"""
Exportador del paquete JSON para el aplicativo web interactivo de la Scorecard.

Lee los artefactos reales entrenados (preprocesador compartido, MLP de PD,
Regresión Logística, resultados de Scorecard, importancia por permutación y,
si existen, los artefactos de LGD/EAD de Parte 2), genera un único archivo
JSON compacto en `docs/app/data/modelo_web.json` y ejecuta una verificación
estricta de paridad numérica usando exactamente la misma lógica matemática que
corre en el navegador.

Uso (desde parte_1/):
    python -m src.exportar_app --config configs/pd.yaml
    python -m src.exportar_app --config configs/pd.yaml --demo
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.preprocesamiento import cargar_config
from src.red_neuronal import MLP
from src.scorecard import prob_a_puntaje


ETIQUETAS_VARIABLES = {
    "loan_amnt": "Monto solicitado (USD)",
    "funded_amnt": "Monto fondeado (USD)",
    "int_rate": "Tasa de interés anual (%)",
    "installment": "Cuota mensual (USD)",
    "annual_inc": "Ingreso anual bruto (USD)",
    "dti": "Relación deuda / ingreso (%)",
    "delinq_2yrs": "Moras +30 días en últimos 2 años",
    "inq_last_6mths": "Consultas crediticias (últimos 6 meses)",
    "open_acc": "Líneas de crédito abiertas",
    "pub_rec": "Registros públicos adversos",
    "revol_bal": "Saldo revolvente actual (USD)",
    "revol_util": "Utilización de crédito revolvente (%)",
    "total_acc": "Total histórico de líneas de crédito",
    "plazo_meses": "Plazo del préstamo (meses)",
    "antiguedad_empleo": "Antigüedad laboral (años)",
    "meses_historial_credito": "Antigüedad del historial (meses)",
    "grade": "Grado crediticio Lending Club",
    "home_ownership": "Tenencia de vivienda",
    "verification_status": "Verificación de ingresos",
    "purpose": "Propósito del préstamo",
    "initial_list_status": "Estado inicial de listado",
}


def redondear_matriz(arr: np.ndarray, decimales: int = 7) -> list:
    return np.round(np.asarray(arr, dtype=float), decimales).tolist()


def redondear_lista(arr: np.ndarray, decimales: int = 7) -> list[float]:
    return [round(float(x), decimales) for x in np.asarray(arr, dtype=float)]


def calcular_cuota_francesa(monto: float, tasa_anual_pct: float, plazo_meses: int) -> float:
    r = (tasa_anual_pct / 100.0) / 12.0
    if r <= 0:
        return round(monto / plazo_meses, 2)
    factor = (1.0 + r) ** plazo_meses
    return round(monto * (r * factor) / (factor - 1.0), 2)


def extraer_preprocesador(prep) -> dict:
    """Extrae los parámetros exactos de SimpleImputer + StandardScaler + OneHotEncoder."""
    cols_num = list(prep.transformers_[0][2])
    cols_cat = list(prep.transformers_[1][2])

    num_pipe = prep.named_transformers_["num"]
    imp_num = num_pipe.named_steps["imputar"]
    esc_num = num_pipe.named_steps["escalar"]

    cat_pipe = prep.named_transformers_["cat"]
    imp_cat = cat_pipe.named_steps["imputar"]
    ohe = cat_pipe.named_steps["codificar"]

    categoricas = []
    for idx, nombre in enumerate(cols_cat):
        todas = [str(c) for c in ohe.categories_[idx]]
        infreq_raw = ohe.infrequent_categories_[idx]
        infrecuentes = [str(c) for c in infreq_raw] if infreq_raw is not None else []
        frecuentes = [c for c in todas if c not in infrecuentes]
        categoricas.append(
            {
                "nombre": nombre,
                "moda": str(imp_cat.statistics_[idx]),
                "frecuentes": frecuentes,
                "infrecuentes": infrecuentes,
                "tiene_infrecuente": infreq_raw is not None,
                "todas_categorias": todas,
            }
        )

    return {
        "numericas": {
            "columnas": cols_num,
            "medianas": [float(v) for v in imp_num.statistics_],
            "indicadores_indices": [int(i) for i in imp_num.indicator_.features_],
            "media": [float(v) for v in esc_num.mean_],
            "escala": [float(v) for v in esc_num.scale_],
        },
        "categoricas": categoricas,
        "entradas_salida": prep.get_feature_names_out().tolist(),
    }


def extraer_capas_mlp(estado: dict) -> list[dict]:
    """Extrae pesos y sesgos de todas las capas lineales de un Sequential ordenado."""
    indices = sorted(
        {
            int(clave.split(".")[1])
            for clave in estado
            if clave.startswith("red.") and clave.endswith(".weight")
        }
    )
    capas = []
    for idx in indices:
        peso = estado[f"red.{idx}.weight"].detach().cpu().numpy()
        sesgo = estado[f"red.{idx}.bias"].detach().cpu().numpy()
        capas.append(
            {
                "W": redondear_matriz(peso),
                "b": redondear_lista(sesgo),
            }
        )
    return capas


def construir_perfiles_ejemplo() -> list[dict]:
    """Tres perfiles realistas en unidades originales del problema."""
    especificaciones = [
        {
            "id": "bajo_riesgo",
            "nombre": "Cliente conservador",
            "descripcion": "Grado A, tasa baja, plazo de 36 meses, alto ingreso y baja utilización de tarjetas.",
            "valores": {
                "loan_amnt": 10000.0,
                "int_rate": 7.5,
                "plazo_meses": 36,
                "grade": "A",
                "purpose": "credit_card",
                "annual_inc": 95000.0,
                "dti": 9.5,
                "home_ownership": "MORTGAGE",
                "verification_status": "Not Verified",
                "antiguedad_empleo": 10.0,
                "revol_util": 22.0,
                "revol_bal": 7500.0,
                "meses_historial_credito": 210.0,
                "open_acc": 11.0,
                "total_acc": 28.0,
                "inq_last_6mths": 0.0,
                "delinq_2yrs": 0.0,
                "pub_rec": 0.0,
                "initial_list_status": "w",
            },
        },
        {
            "id": "perfil_medio",
            "nombre": "Perfil promedio",
            "descripcion": "Consolidación de deuda a 36 meses en grado C con indicadores intermedios de la cartera.",
            "valores": {
                "loan_amnt": 14000.0,
                "int_rate": 13.5,
                "plazo_meses": 36,
                "grade": "C",
                "purpose": "debt_consolidation",
                "annual_inc": 65000.0,
                "dti": 17.5,
                "home_ownership": "RENT",
                "verification_status": "Source Verified",
                "antiguedad_empleo": 6.0,
                "revol_util": 55.0,
                "revol_bal": 12000.0,
                "meses_historial_credito": 160.0,
                "open_acc": 10.0,
                "total_acc": 23.0,
                "inq_last_6mths": 1.0,
                "delinq_2yrs": 0.0,
                "pub_rec": 0.0,
                "initial_list_status": "f",
            },
        },
        {
            "id": "alto_riesgo",
            "nombre": "Perfil de alto riesgo",
            "descripcion": "Pequeña empresa a 60 meses en grado F, DTI elevado y sin dato reportado de antigüedad laboral.",
            "valores": {
                "loan_amnt": 26000.0,
                "int_rate": 22.5,
                "plazo_meses": 60,
                "grade": "F",
                "purpose": "small_business",
                "annual_inc": 38000.0,
                "dti": 31.0,
                "home_ownership": "RENT",
                "verification_status": "Verified",
                "antiguedad_empleo": None,
                "revol_util": 88.0,
                "revol_bal": 21000.0,
                "meses_historial_credito": 84.0,
                "open_acc": 16.0,
                "total_acc": 20.0,
                "inq_last_6mths": 4.0,
                "delinq_2yrs": 2.0,
                "pub_rec": 1.0,
                "initial_list_status": "f",
            },
        },
    ]
    for perfil in especificaciones:
        v = perfil["valores"]
        v["funded_amnt"] = v["loan_amnt"]
        v["installment"] = calcular_cuota_francesa(v["loan_amnt"], v["int_rate"], int(v["plazo_meses"]))
    return especificaciones


def construir_mapa_variables(entradas: list[str], cols_num: list[str], cols_cat: list[str]) -> list[dict]:
    """Agrupa las 54 columnas codificadas por su variable original (21 grupos)."""
    mapa = []
    for var in cols_num:
        indices = [
            i for i, col in enumerate(entradas)
            if col == f"num__{var}" or col == f"num__missingindicator_{var}"
        ]
        mapa.append({"variable": var, "etiqueta": ETIQUETAS_VARIABLES.get(var, var), "indices": indices})
    for var in cols_cat:
        prefijo = f"cat__{var}_"
        indices = [i for i, col in enumerate(entradas) if col.startswith(prefijo)]
        mapa.append({"variable": var, "etiqueta": ETIQUETAS_VARIABLES.get(var, var), "indices": indices})
    return mapa


# =============================================================================
# Motor puro de verificación (misma lógica matemática exacta que app.js)
# =============================================================================
def _preprocesar_puro(perfil: dict, prep_json: dict) -> np.ndarray:
    num = prep_json["numericas"]
    cols_num = num["columnas"]
    medianas = num["medianas"]
    ind_indices = num["indicadores_indices"]
    media = num["media"]
    escala = num["escala"]

    vector_num = []
    faltantes = []
    for i, col in enumerate(cols_num):
        val = perfil.get(col)
        es_faltante = val is None or (isinstance(val, float) and math.isnan(val))
        faltantes.append(es_faltante)
        vector_num.append(medianas[i] if es_faltante else float(val))

    for idx in ind_indices:
        vector_num.append(1.0 if faltantes[idx] else 0.0)

    salida = [(vector_num[k] - media[k]) / escala[k] for k in range(len(vector_num))]

    for cat in prep_json["categoricas"]:
        raw = perfil.get(cat["nombre"])
        val = cat["moda"] if (raw is None or raw == "") else str(raw)
        frecuentes = cat["frecuentes"]
        bits = [1.0 if val == c else 0.0 for c in frecuentes]
        if cat["tiene_infrecuente"]:
            bits.append(0.0 if val in frecuentes else 1.0)
        salida.extend(bits)

    return np.asarray(salida, dtype=float)


def _predecir_mlp_puro(X: np.ndarray, modelo_json: dict) -> np.ndarray:
    h = np.asarray(X, dtype=float)
    act = modelo_json["activacion"]
    capas = modelo_json["capas"]
    for idx, capa in enumerate(capas):
        W = np.asarray(capa["W"], dtype=float)
        b = np.asarray(capa["b"], dtype=float)
        h = h @ W.T + b
        if idx < len(capas) - 1:
            if act == "relu":
                h = np.maximum(0.0, h)
            elif act == "tanh":
                h = np.tanh(h)
            elif act == "elu":
                h = np.where(h >= 0.0, h, np.exp(h) - 1.0)
            else:
                raise ValueError(f"Activación no soportada: {act}")
    z = np.clip(h.ravel(), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _predecir_lr_puro(X: np.ndarray, lr_json: dict) -> np.ndarray:
    coef = np.asarray(lr_json["coeficientes"], dtype=float)
    z = np.clip(X @ coef + float(lr_json["intercepto"]), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def verificar_paridad_numerica(
    data: dict,
    prep_sklearn,
    ckpt_pd: dict,
    paq_lr: dict,
    df_prueba: pd.DataFrame,
    ckpt_lgd: dict | None,
    n_muestras: int,
    semilla: int,
) -> dict:
    """Verifica que la reconstrucción pura desde JSON reproduce los modelos nativos."""
    # 1. Paridad de preprocesamiento en los 3 presets + 1 caso con categorías infrecuentes
    cols_entrada = prep_sklearn.transformers_[0][2] + prep_sklearn.transformers_[1][2]
    casos = [p["valores"] for p in data["perfiles_ejemplo"]]
    caso_infrec = dict(casos[1])
    caso_infrec["home_ownership"] = "OTHER"
    caso_infrec["purpose"] = "educational"
    caso_infrec["delinq_2yrs"] = None
    casos.append(caso_infrec)

    df_casos = pd.DataFrame([{c: caso.get(c, np.nan) for c in cols_entrada} for caso in casos])
    df_casos = df_casos.astype({c: "float64" for c in prep_sklearn.transformers_[0][2]})
    X_sklearn = np.asarray(prep_sklearn.transform(df_casos), dtype=float)
    X_puro = np.vstack([_preprocesar_puro(caso, data["preprocesador"]) for caso in casos])
    err_prep = float(np.max(np.abs(X_sklearn - X_puro)))
    if err_prep >= 1e-6:
        raise RuntimeError(f"Paridad fallida en preprocesador: error máx {err_prep:.2e}")

    # 2. Paridad en muestras reales del conjunto de prueba
    rng = np.random.default_rng(semilla)
    n = min(n_muestras, len(df_prueba))
    idx = rng.choice(len(df_prueba), size=n, replace=False)
    entradas = data["preprocesador"]["entradas_salida"]
    X_sub = df_prueba.iloc[idx][entradas].to_numpy(dtype=np.float32)

    # MLP PD
    hp = ckpt_pd["hiperparametros"]
    mlp_pd_torch = MLP(len(entradas), hp["capas_ocultas"], hp["activacion"], hp["dropout"])
    mlp_pd_torch.load_state_dict(ckpt_pd["estado"])
    mlp_pd_torch.eval()
    with torch.no_grad():
        pd_torch = torch.sigmoid(mlp_pd_torch(torch.tensor(X_sub))).numpy().astype(float)
    pd_json = _predecir_mlp_puro(X_sub.astype(float), data["mlp_pd"])
    err_mlp_pd = float(np.max(np.abs(pd_torch - pd_json)))
    if err_mlp_pd >= 1e-5:
        raise RuntimeError(f"Paridad fallida en MLP PD: error máx {err_mlp_pd:.2e}")

    # Regresión Logística y Scorecard aditiva
    pd_lr_sk = paq_lr["modelo"].predict_proba(X_sub.astype(float))[:, 1]
    pd_lr_json = _predecir_lr_puro(X_sub.astype(float), data["regresion_logistica"])
    err_lr = float(np.max(np.abs(pd_lr_sk - pd_lr_json)))

    sc = data["scorecard"]
    sc_formula = prob_a_puntaje(pd_lr_sk, sc["A"], sc["B"])
    sc_aditivo = float(sc["puntaje_base_aditivo_logistica"]) + X_sub.astype(float) @ np.asarray(
        data["regresion_logistica"]["pesos_puntos"], dtype=float
    )
    err_sc = float(np.max(np.abs(sc_formula - sc_aditivo)))
    if err_lr >= 1e-5 or err_sc >= 1e-5:
        raise RuntimeError(f"Paridad fallida en LR/Scorecard: lr={err_lr:.2e}, sc={err_sc:.2e}")

    # LGD Parte 2 (opcional)
    err_lgd = None
    if ckpt_lgd is not None and data.get("lgd_parte2") is not None:
        hp2 = ckpt_lgd["hiperparametros"]
        mlp_lgd_torch = MLP(len(entradas), hp2["capas_ocultas"], hp2["activacion"], hp2["dropout"])
        mlp_lgd_torch.load_state_dict(ckpt_lgd["estado"])
        mlp_lgd_torch.eval()
        with torch.no_grad():
            lgd_torch = torch.sigmoid(mlp_lgd_torch(torch.tensor(X_sub))).numpy().astype(float)
        lgd_json = _predecir_mlp_puro(X_sub.astype(float), data["lgd_parte2"])
        err_lgd = float(np.max(np.abs(lgd_torch - lgd_json)))
        if err_lgd >= 1e-5:
            raise RuntimeError(f"Paridad fallida en MLP LGD: error máx {err_lgd:.2e}")

    return {
        "max_error_preprocesador": err_prep,
        "max_error_mlp_pd": err_mlp_pd,
        "max_error_regresion_logistica": err_lr,
        "max_error_scorecard_aditiva": err_sc,
        "max_error_mlp_lgd": err_lgd,
        "muestras_probadas": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta modelos y Scorecard a JSON para la app web")
    parser.add_argument("--config", default="configs/pd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    parser.add_argument("--salida", default=None, help="Ruta de salida JSON (sobrescribe app_web.salida_json)")
    parser.add_argument("--demo", action="store_true", help="Lee artefactos desde subcarpetas demo/")
    args = parser.parse_args()

    config = cargar_config(args.config, args.set)
    app_cfg = config.get("app_web", {})
    sub = "demo" if args.demo else ""

    ruta_salida = Path(
        args.salida
        if args.salida
        else (app_cfg.get("salida_json", "../docs/app/data/modelo_web.json")
              if not args.demo
              else "../docs/app/data/demo/modelo_web.json")
    )
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    # Preprocesador
    ruta_prep = Path(config["rutas"]["salida"]) / "preprocesador.joblib"
    if not ruta_prep.exists():
        ruta_prep = Path(config["rutas"]["preprocesador_compartido"])
    prep_sklearn = joblib.load(ruta_prep)
    prep_json = extraer_preprocesador(prep_sklearn)
    entradas = prep_json["entradas_salida"]

    # Rutas de resultados
    dir_lr = Path(config["regresion_logistica"]["resultados"]) / sub
    dir_mlp = Path(config["red_neuronal"]["resultados"]) / sub
    dir_eval = Path(config["evaluacion"]["resultados"]) / sub
    dir_sc = Path(config["scorecard"]["resultados"]) / sub
    dir_int = Path(config["interpretabilidad"]["resultados"]) / sub

    # MLP PD
    ckpt_pd = torch.load(dir_mlp / "modelo_red.pt", map_location="cpu", weights_only=False)
    if ckpt_pd["entradas"] != entradas:
        raise ValueError("Las columnas del MLP de PD no coinciden con las del preprocesador.")
    hp_pd = ckpt_pd["hiperparametros"]
    mlp_pd_json = {
        "activacion": hp_pd["activacion"],
        "capas_ocultas": [int(c) for c in hp_pd["capas_ocultas"]],
        "dropout": float(hp_pd["dropout"]),
        "optimizador": str(hp_pd["optimizador"]),
        "parametros": int(ckpt_pd.get("parametros", 0)),
        "semilla": int(ckpt_pd.get("semilla", config["red_neuronal"]["entrenar"]["semilla"])),
        "mejor_epoca": int(ckpt_pd.get("mejor_epoca", 0)),
        "capas": extraer_capas_mlp(ckpt_pd["estado"]),
    }

    # Regresión Logística
    paq_lr = joblib.load(dir_lr / "modelo_logistica.joblib")
    if paq_lr["entradas"] != entradas:
        raise ValueError("Las columnas de la Regresión Logística no coinciden con las del preprocesador.")
    modelo_lr = paq_lr["modelo"]
    coefs = modelo_lr.coef_.ravel().astype(float)
    intercepto = float(modelo_lr.intercept_[0])

    # Scorecard y población de prueba
    resumen_sc = json.loads((dir_sc / "resumen_scorecard.json").read_text(encoding="utf-8"))
    bandas_df = pd.read_csv(dir_sc / "resumen_bandas_scorecard.csv")
    puntajes_df = pd.read_parquet(dir_sc / "puntajes_prueba.parquet")
    df_prueba_raw = pd.read_parquet(Path(config["rutas"]["salida"]) / "pd_prueba.parquet")

    params_sc = resumen_sc["parametros"]
    b_sc = float(params_sc["B"])
    pesos_puntos = redondear_lista(-b_sc * coefs)

    # Media de las 54 entradas en el conjunto de prueba (para descomposición aditiva centrada en la cartera promedio)
    media_cartera_vec = df_prueba_raw[entradas].to_numpy(dtype=float).mean(axis=0)
    puntaje_lr_cartera_media = float(
        resumen_sc["puntaje_base_aditivo_logistica"] + float(media_cartera_vec @ (-b_sc * coefs))
    )

    p_red = puntajes_df["puntaje_red"].to_numpy(dtype=float)
    p_lr = puntajes_df["puntaje_logistica"].to_numpy(dtype=float)
    malos = puntajes_df["malo"].to_numpy(dtype=int)

    n_bins = int(app_cfg.get("bins_histograma", 40))
    lo = float(np.quantile(p_red, 0.002))
    hi = float(np.quantile(p_red, 0.998))
    bordes = np.linspace(lo, hi, n_bins + 1)
    buenos_arr = p_red[malos == 0]
    malos_arr = p_red[malos == 1]
    dens_b, _ = np.histogram(buenos_arr, bins=bordes, density=True)
    dens_m, _ = np.histogram(malos_arr, bins=bordes, density=True)
    cnt_b, _ = np.histogram(buenos_arr, bins=bordes)
    cnt_m, _ = np.histogram(malos_arr, bins=bordes)

    # Evaluación
    resumen_eval = json.loads((dir_eval / "resumen_evaluacion.json").read_text(encoding="utf-8"))

    # Interpretabilidad
    df_imp = pd.read_csv(dir_int / "importancia_permutacion.csv")
    importancia_json = [
        {
            "variable": str(fila["variable"]),
            "etiqueta": ETIQUETAS_VARIABLES.get(str(fila["variable"]), str(fila["variable"])),
            "mlp_delta_auc": round(float(fila["mlp_delta_auc_mean"]), 6),
            "lr_delta_auc": round(float(fila["lr_delta_auc_mean"]), 6),
        }
        for _, fila in df_imp.iterrows()
    ]

    # Parte 2 opcional (LGD + EAD)
    ruta_p2_red = Path(app_cfg.get("ruta_parte_2_red", "../parte_2/results/red/modelo_red.pt"))
    ruta_p2_perdida = Path(app_cfg.get("ruta_parte_2_perdida", "../parte_2/results/perdida_esperada/resumen.json"))
    ckpt_lgd = None
    lgd_json = None
    if not args.demo and ruta_p2_red.exists():
        ckpt_lgd = torch.load(ruta_p2_red, map_location="cpu", weights_only=False)
        if ckpt_lgd.get("entradas") == entradas:
            hp2 = ckpt_lgd["hiperparametros"]
            res_p2 = json.loads(ruta_p2_perdida.read_text(encoding="utf-8")) if ruta_p2_perdida.exists() else {}
            lgd_json = {
                "activacion": hp2["activacion"],
                "capas_ocultas": [int(c) for c in hp2["capas_ocultas"]],
                "semilla": int(ckpt_lgd.get("semilla", 0)),
                "capas": extraer_capas_mlp(ckpt_lgd["estado"]),
                "ead_fraccion_por_plazo": res_p2.get("ead", {}).get("fraccion_por_plazo", {"36": 0.601, "60": 0.749, "todos": 0.649}),
                "lgd_medio_prueba": round(float(res_p2.get("lgd_predicho_medio", 0.9226)), 5),
                "tasa_perdida_esperada_cartera": round(
                    float(res_p2["perdida_esperada_total"] / res_p2["monto_prestado"]), 5
                ) if res_p2.get("monto_prestado") else None,
            }

    cols_num = prep_json["numericas"]["columnas"]
    cols_cat = [c["nombre"] for c in prep_json["categoricas"]]

    paquete = {
        "meta": {
            "version_esquema": 1,
            "fecha_exportacion": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "demo": bool(args.demo),
            "n_entradas": len(entradas),
            "n_variables_originales": len(cols_num) + len(cols_cat),
            "n_prueba": int(len(puntajes_df)),
        },
        "preprocesador": prep_json,
        "mapa_variables": construir_mapa_variables(entradas, cols_num, cols_cat),
        "mlp_pd": mlp_pd_json,
        "regresion_logistica": {
            "C": float(paq_lr["C"]),
            "semilla": int(paq_lr["semilla"]),
            "intercepto": round(intercepto, 7),
            "coeficientes": redondear_lista(coefs),
            "pesos_puntos": pesos_puntos,
            "media_cartera_x": redondear_lista(media_cartera_vec, 6),
            "puntaje_cartera_media": round(puntaje_lr_cartera_media, 4),
        },
        "scorecard": {
            "P0": float(params_sc["P0"]),
            "odds_referencia": float(params_sc["odds_referencia"]),
            "pdo": float(params_sc["pdo"]),
            "A": float(params_sc["A"]),
            "B": float(params_sc["B"]),
            "puntaje_base_aditivo_logistica": float(resumen_sc["puntaje_base_aditivo_logistica"]),
            "percentiles_101": [round(float(np.percentile(p_red, q)), 4) for q in range(101)],
            "percentiles_101_lr": [round(float(np.percentile(p_lr, q)), 4) for q in range(101)],
            "histograma": {
                "bordes": [round(float(v), 4) for v in bordes],
                "densidad_buenos": [round(float(v), 6) for v in dens_b],
                "densidad_malos": [round(float(v), 6) for v in dens_m],
                "conteo_buenos": [int(v) for v in cnt_b],
                "conteo_malos": [int(v) for v in cnt_m],
                "mediana_buenos": round(float(np.median(buenos_arr)), 2),
                "mediana_malos": round(float(np.median(malos_arr)), 2),
                "media_global": round(float(np.mean(p_red)), 2),
                "tasa_malos_global": round(float(np.mean(malos)), 5),
                "n_prueba": int(len(puntajes_df)),
            },
            "bandas": [
                {
                    "banda": int(r["banda"]),
                    "creditos": int(r["creditos"]),
                    "malos": int(r["malos"]),
                    "puntaje_min": round(float(r["puntaje_min"]), 2),
                    "puntaje_max": round(float(r["puntaje_max"]), 2),
                    "puntaje_medio": round(float(r["puntaje_medio"]), 2),
                    "pd_media_predicha": round(float(r["pd_media_predicha"]), 5),
                    "tasa_malos_observada": round(float(r["tasa_malos_observada"]), 5),
                }
                for _, r in bandas_df.iterrows()
            ],
        },
        "evaluacion_prueba": {
            "red_neuronal": {k: round(float(v), 5) for k, v in resumen_eval["metricas"]["prueba"]["red_neuronal"].items()},
            "regresion_logistica": {k: round(float(v), 5) for k, v in resumen_eval["metricas"]["prueba"]["regresion_logistica"].items()},
        },
        "importancia_permutacion": importancia_json,
        "lgd_parte2": lgd_json,
        "perfiles_ejemplo": construir_perfiles_ejemplo(),
    }

    # Serializar -> recargar para verificar EXACTAMENTE lo que recibirá app.js
    texto_json = json.dumps(paquete, ensure_ascii=False, separators=(",", ":"))
    paquete_recargado = json.loads(texto_json)

    verif = verificar_paridad_numerica(
        data=paquete_recargado,
        prep_sklearn=prep_sklearn,
        ckpt_pd=ckpt_pd,
        paq_lr=paq_lr,
        df_prueba=df_prueba_raw,
        ckpt_lgd=ckpt_lgd,
        n_muestras=int(app_cfg.get("muestras_verificacion", 200)),
        semilla=int(config.get("semilla", 2026)),
    )

    ruta_salida.write_text(texto_json, encoding="utf-8")
    ruta_salida_js = ruta_salida.with_suffix(".js")
    ruta_salida_js.write_text(
        "// Generado automáticamente por parte_1/src/exportar_app.py — NO editar manualmente.\n"
        f"globalThis.__MODELO_WEB__ = {texto_json};\n",
        encoding="utf-8",
    )
    tamano_kb = ruta_salida.stat().st_size / 1024.0
    tamano_js_kb = ruta_salida_js.stat().st_size / 1024.0

    print(f"=== EXPORTACIÓN APP WEB SCORECARD{' [DEMO]' if args.demo else ''} ===")
    print(f"Archivo JSON     : {ruta_salida.resolve()} ({tamano_kb:.1f} KB)")
    print(f"Archivo JS (file): {ruta_salida_js.resolve()} ({tamano_js_kb:.1f} KB)")
    print(f"Entradas totales : {paquete['meta']['n_entradas']} ({paquete['meta']['n_variables_originales']} variables)")
    print(f"MLP PD           : capas={mlp_pd_json['capas_ocultas']}, act={mlp_pd_json['activacion']}, semilla={mlp_pd_json['semilla']}")
    print(f"LGD Parte 2      : {'incluido (capas=' + str(lgd_json['capas_ocultas']) + ', act=' + lgd_json['activacion'] + ')' if lgd_json else 'no incluido'}")
    print(f"Paridad ({verif['muestras_probadas']} muestras prueba):")
    print(f"  Preprocesador  : error máx = {verif['max_error_preprocesador']:.2e} (< 1e-6)")
    print(f"  MLP PD         : error máx = {verif['max_error_mlp_pd']:.2e} (< 1e-5)")
    print(f"  Reg. Logística : error máx = {verif['max_error_regresion_logistica']:.2e} (< 1e-5)")
    print(f"  Scorecard adit.: error máx = {verif['max_error_scorecard_aditiva']:.2e} (< 1e-5)")
    if verif["max_error_mlp_lgd"] is not None:
        print(f"  MLP LGD (P2)   : error máx = {verif['max_error_mlp_lgd']:.2e} (< 1e-5)")


if __name__ == "__main__":
    main()
