"""
Perceptrón multicapa (MLP) en PyTorch para Probabilidad de Incumplimiento (PD).

Modos de ejecución (desde parte_1/):
    python -m src.red_neuronal --config configs/pd.yaml --modo entrenar
    python -m src.red_neuronal --config configs/pd.yaml --modo busqueda
    python -m src.red_neuronal --config configs/pd.yaml --modo semillas

Agregar --demo reduce datos, épocas, configuraciones y semillas para correr en
menos de 2 minutos sin modificar los resultados finales ni intercambio/pd_prueba.csv.
Cualquier hiperparámetro se puede modificar sin editar código usando --set:
    --set red_neuronal.entrenar.dropout=0.25 --set red_neuronal.entrenar.tasa_aprendizaje=0.005
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.preprocesamiento import cargar_config
from src.regresion_logistica import CONJUNTOS, PREFIJOS_ENTRADA, calcular_metricas

CLAVES_HP = [
    "capas_ocultas",
    "activacion",
    "dropout",
    "optimizador",
    "tasa_aprendizaje",
    "penalizacion_l2",
    "tamano_lote",
]
ACTIVACIONES = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}

# Paleta editorial compartida con Parte 2
AZUL, NARANJA = "#2a78d6", "#eb6834"
TEXTO, TEXTO_2, FONDO, GUIA = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d4"


def _estilo(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(FONDO)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GUIA)
    ax.spines["bottom"].set_color(GUIA)
    ax.tick_params(colors=TEXTO_2, labelsize=9)
    ax.grid(axis=grid_axis, color=GUIA, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


# =============================================================================
# Modelo y utilidades de entrenamiento
# =============================================================================
class MLP(nn.Module):
    """Perceptrón multicapa para clasificación binaria que devuelve logits z."""

    def __init__(self, n_entradas: int, capas_ocultas: list[int], activacion: str, dropout: float):
        super().__init__()
        capas: list[nn.Module] = []
        ancho = n_entradas
        for n in capas_ocultas:
            capas += [nn.Linear(ancho, int(n)), ACTIVACIONES[activacion](), nn.Dropout(float(dropout))]
            ancho = int(n)
        capas.append(nn.Linear(ancho, 1))
        self.red = nn.Sequential(*capas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x).squeeze(-1)


def contar_parametros(modelo: nn.Module) -> int:
    return int(sum(p.numel() for p in modelo.parameters() if p.requires_grad))


def crear_optimizador(nombre: str, parametros, lr: float, l2: float):
    if nombre == "adam":
        return torch.optim.Adam(parametros, lr=lr, weight_decay=l2)
    if nombre == "sgd":
        return torch.optim.SGD(parametros, lr=lr, momentum=0.9, weight_decay=l2)
    raise ValueError(f"Optimizador desconocido: {nombre}")


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def entrenar_modelo(
    hp: dict,
    datos: dict[str, tuple[torch.Tensor, torch.Tensor]],
    semilla: int,
    epocas_max: int,
    paciencia: int,
) -> dict:
    """Entrena con mini-lotes y parada temprana sobre la pérdida BCE de validación."""
    fijar_semilla(semilla)
    Xtr, ytr = datos["entrenamiento"]
    Xva, yva = datos["validacion"]
    modelo = MLP(Xtr.shape[1], hp["capas_ocultas"], hp["activacion"], hp["dropout"])
    criterio = nn.BCEWithLogitsLoss()
    opt = crear_optimizador(
        hp["optimizador"],
        modelo.parameters(),
        float(hp["tasa_aprendizaje"]),
        float(hp["penalizacion_l2"]),
    )
    generador = torch.Generator().manual_seed(semilla)
    tamano_lote = int(hp["tamano_lote"])

    mejor_val = float("inf")
    mejor_estado = copy.deepcopy(modelo.state_dict())
    mejor_epoca = 0
    sin_mejorar = 0
    historial = []
    t0 = time.perf_counter()

    for epoca in range(1, int(epocas_max) + 1):
        modelo.train()
        orden = torch.randperm(len(ytr), generator=generador)
        suma, n = 0.0, 0
        for i in range(0, len(ytr), tamano_lote):
            idx = orden[i:i + tamano_lote]
            opt.zero_grad()
            perdida = criterio(modelo(Xtr[idx]), ytr[idx])
            perdida.backward()
            opt.step()
            suma += float(perdida.item()) * len(idx)
            n += len(idx)

        modelo.eval()
        with torch.no_grad():
            val_loss = float(criterio(modelo(Xva), yva).item())
        historial.append({
            "epoca": epoca,
            "perdida_entrenamiento": suma / n,
            "perdida_validacion": val_loss,
        })

        if np.isfinite(val_loss) and val_loss < mejor_val - 1e-6:
            mejor_val = val_loss
            mejor_epoca = epoca
            sin_mejorar = 0
            mejor_estado = copy.deepcopy(modelo.state_dict())
        else:
            sin_mejorar += 1
            if sin_mejorar >= int(paciencia):
                break

    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    return {
        "modelo": modelo,
        "historial": pd.DataFrame(historial),
        "mejor_epoca": int(mejor_epoca),
        "epocas_corridas": int(len(historial)),
        "segundos": float(time.perf_counter() - t0),
    }


def predecir(modelo: nn.Module, X: torch.Tensor) -> np.ndarray:
    modelo.eval()
    with torch.no_grad():
        return torch.sigmoid(modelo(X)).numpy().astype(float)


def evaluar(modelo: nn.Module, datos: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> dict[str, dict]:
    return {
        c: calcular_metricas(datos[c][1].numpy().astype(int), predecir(modelo, datos[c][0]))
        for c in CONJUNTOS
    }


# =============================================================================
# Carga de datos y figuras
# =============================================================================
def preparar_datos(config: dict, demo: bool) -> tuple[dict, list[str], pd.DataFrame, pd.DataFrame]:
    carpeta = Path(config["rutas"]["salida"])
    tablas = {c: pd.read_parquet(carpeta / f"pd_{c}.parquet") for c in CONJUNTOS}
    if demo:
        frac = float(config["red_neuronal"]["demo"]["fraccion_entrenamiento"])
        tablas["entrenamiento"] = tablas["entrenamiento"].sample(
            frac=frac, random_state=int(config["semilla"])
        ).reset_index(drop=True)
    entradas = [c for c in tablas["entrenamiento"].columns if c.startswith(PREFIJOS_ENTRADA)]
    if not entradas:
        raise ValueError("No se encontraron columnas de entrada num__ / cat__.")
    datos = {
        c: (
            torch.tensor(tablas[c][entradas].to_numpy(), dtype=torch.float32),
            torch.tensor(tablas[c]["malo"].astype(int).to_numpy(), dtype=torch.float32),
        )
        for c in CONJUNTOS
    }
    val_meta = tablas["validacion"][["id", "malo"]].copy()
    prueba_meta = tablas["prueba"][["id", "malo"]].copy()
    return datos, entradas, val_meta, prueba_meta


def figura_historial(historial: pd.DataFrame, mejor_epoca: int, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=FONDO)
    _estilo(ax)
    ax.plot(historial["epoca"], historial["perdida_entrenamiento"], color=AZUL, linewidth=2.0, label="Entrenamiento")
    ax.plot(historial["epoca"], historial["perdida_validacion"], color=NARANJA, linewidth=2.0, label="Validación (2013)")
    if mejor_epoca > 0:
        ax.axvline(mejor_epoca, color=TEXTO_2, linestyle="--", linewidth=1.2, label=f"Mejor época ({mejor_epoca})")
    ax.set_xlabel("Época", color=TEXTO, fontsize=10)
    ax.set_ylabel("Entropía cruzada binaria (BCE)", color=TEXTO, fontsize=10)
    ax.set_title("Curva de aprendizaje del MLP (parada temprana en validación)", color=TEXTO, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXTO_2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, facecolor=FONDO)
    plt.close(fig)


def figura_busqueda(tabla: pd.DataFrame, metrica: str, ruta: Path) -> None:
    col_val = f"val_{metrica}"
    asc = metrica in {"log_loss", "brier"}
    ord_df = tabla.sort_values(col_val, ascending=asc).reset_index(drop=True)
    mejor = ord_df.iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=FONDO)
    ax0, ax1 = axes[0], axes[1]
    _estilo(ax0)
    _estilo(ax1, grid_axis="x")

    for opt_name, color, marker in [("adam", AZUL, "o"), ("sgd", NARANJA, "s")]:
        sub = ord_df[ord_df["optimizador"] == opt_name]
        if len(sub):
            ax0.scatter(sub["parametros"], sub[col_val], color=color, marker=marker, alpha=0.78, s=55,
                        label=f"Optimizador {opt_name.upper()}")
    ax0.scatter([mejor["parametros"]], [mejor[col_val]], s=140, facecolors="none", edgecolors=TEXTO,
                linewidths=2.0, zorder=5, label=f"Mejor (#{int(mejor['configuracion'])})")
    ax0.set_xscale("log")
    ax0.set_xlabel("Parámetros entrenables (escala log)", color=TEXTO, fontsize=9.5)
    ax0.set_ylabel(f"Validación ({metrica.upper()})", color=TEXTO, fontsize=9.5)
    ax0.set_title("A. Complejidad vs. desempeño en validación", color=TEXTO, fontsize=10.5, fontweight="bold")
    ax0.legend(frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8)

    top_n = ord_df.head(min(10, len(ord_df))).iloc[::-1]
    etiquetas = [
        f"#{int(r['configuracion'])} {r['capas_ocultas']} {r['activacion']} {r['optimizador']} d={r['dropout']}"
        for _, r in top_n.iterrows()
    ]
    colores_b = [NARANJA if int(r["configuracion"]) == int(mejor["configuracion"]) else AZUL for _, r in top_n.iterrows()]
    bars = ax1.barh(etiquetas, top_n[col_val], color=colores_b)
    xmin = float(top_n[col_val].min())
    xmax = float(top_n[col_val].max())
    margen = max((xmax - xmin) * 0.25, 0.002)
    ax1.set_xlim(max(0.0, xmin - margen), xmax + margen * 1.4)
    for b, val in zip(bars, top_n[col_val]):
        ax1.text(b.get_width() + margen * 0.08, b.get_y() + b.get_height() / 2, f"{val:.4f}",
                 va="center", fontsize=8, color=TEXTO)
    ax1.set_xlabel(f"Validación ({metrica.upper()})", color=TEXTO, fontsize=9.5)
    ax1.set_title(f"B. Top {len(top_n)} configuraciones evaluadas", color=TEXTO, fontsize=10.5, fontweight="bold")

    fig.suptitle("Búsqueda Aleatoria de Hiperparámetros — MLP (Parte 1: PD)", color=TEXTO, fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ruta, dpi=150, facecolor=FONDO)
    plt.close(fig)


# =============================================================================
# Guardado y modos
# =============================================================================
def guardar_salidas_modelo(
    resultado: dict,
    hp: dict,
    entradas: list[str],
    semilla: int,
    datos: dict,
    val_meta: pd.DataFrame,
    prueba_meta: pd.DataFrame,
    resultados: Path,
    figuras: Path,
) -> None:
    modelo = resultado["modelo"]
    n_par = contar_parametros(modelo)
    torch.save(
        {
            "estado": modelo.state_dict(),
            "hiperparametros": hp,
            "entradas": entradas,
            "semilla": int(semilla),
            "mejor_epoca": resultado["mejor_epoca"],
            "parametros": n_par,
        },
        resultados / "modelo_red.pt",
    )
    resultado["historial"].to_csv(resultados / "historial.csv", index=False)
    figura_historial(resultado["historial"], resultado["mejor_epoca"], figuras / "historial_perdida.png")

    pred_val = predecir(modelo, datos["validacion"][0])
    pred_pru = predecir(modelo, datos["prueba"][0])
    pd.DataFrame({
        "id": val_meta["id"].to_numpy(),
        "malo": val_meta["malo"].astype(int).to_numpy(),
        "pd_red": pred_val,
    }).to_parquet(resultados / "predicciones_validacion.parquet", index=False)
    pd.DataFrame({
        "id": prueba_meta["id"].to_numpy(),
        "malo": prueba_meta["malo"].astype(int).to_numpy(),
        "pd_red": pred_pru,
    }).to_parquet(resultados / "predicciones_prueba.parquet", index=False)


def _filas_metricas(evaluacion: dict, extra: dict) -> list[dict]:
    return [{**extra, "conjunto": c, **m} for c, m in evaluacion.items()]


def modo_entrenar(config, datos, entradas, val_meta, prueba_meta, resultados, figuras, demo) -> pd.DataFrame:
    cfg = config["red_neuronal"]["entrenar"]
    hp = {k: cfg[k] for k in CLAVES_HP}
    epocas = int(config["red_neuronal"]["demo"]["epocas_max"] if demo else cfg["epocas_max"])
    res = entrenar_modelo(hp, datos, int(cfg["semilla"]), epocas, int(cfg["paciencia"]))
    n_par = contar_parametros(res["modelo"])
    tabla = pd.DataFrame(
        _filas_metricas(
            evaluar(res["modelo"], datos),
            {
                "semilla": int(cfg["semilla"]),
                "parametros": n_par,
                "mejor_epoca": res["mejor_epoca"],
                "epocas_corridas": res["epocas_corridas"],
                "segundos": res["segundos"],
            },
        )
    )
    tabla.to_csv(resultados / "metricas_entrenar.csv", index=False)
    guardar_salidas_modelo(res, hp, entradas, int(cfg["semilla"]), datos, val_meta, prueba_meta, resultados, figuras)
    return tabla


def muestrear_configuracion(espacio: dict, rng: np.random.Generator) -> dict:
    hp = {}
    for clave, opciones in espacio.items():
        if isinstance(opciones, dict) and "log_uniforme" in opciones:
            lo, hi = opciones["log_uniforme"]
            hp[clave] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            hp[clave] = opciones[int(rng.integers(len(opciones)))]
    return hp


def modo_busqueda(config, datos, resultados, figuras, demo) -> tuple[pd.DataFrame, dict]:
    cfg = config["red_neuronal"]["busqueda"]
    n_conf = int(config["red_neuronal"]["demo"]["configuraciones"] if demo else cfg["configuraciones"])
    epocas = int(config["red_neuronal"]["demo"]["epocas_max"] if demo else cfg["epocas_max"])
    metrica = config["red_neuronal"]["metrica_seleccion"]
    if metrica not in {"auc", "log_loss", "brier"}:
        raise ValueError(f"metrica_seleccion desconocida: {metrica}")

    rng = np.random.default_rng(int(cfg["semilla"]))
    filas = []
    t0 = time.perf_counter()

    for i in range(1, n_conf + 1):
        hp = muestrear_configuracion(cfg["espacio"], rng)
        res = entrenar_modelo(hp, datos, int(cfg["semilla"]), epocas, int(cfg["paciencia"]))
        # Evaluación estrictamente sobre entrenamiento y validación (prueba no interviene en la búsqueda)
        ev_tr = calcular_metricas(datos["entrenamiento"][1].numpy().astype(int), predecir(res["modelo"], datos["entrenamiento"][0]))
        ev_va = calcular_metricas(datos["validacion"][1].numpy().astype(int), predecir(res["modelo"], datos["validacion"][0]))
        fila = {
            "configuracion": i,
            **{k: hp[k] for k in CLAVES_HP},
            "parametros": contar_parametros(res["modelo"]),
            "mejor_epoca": res["mejor_epoca"],
            "epocas_corridas": res["epocas_corridas"],
            "segundos": res["segundos"],
            "entr_auc": ev_tr["auc"],
            "entr_log_loss": ev_tr["log_loss"],
            "val_auc": ev_va["auc"],
            "val_gini": ev_va["gini"],
            "val_ks": ev_va["ks"],
            "val_brier": ev_va["brier"],
            "val_log_loss": ev_va["log_loss"],
        }
        filas.append(fila)
        print(
            f"  [{i}/{n_conf}] val_{metrica}={fila['val_' + metrica]:.4f}  "
            f"{hp['capas_ocultas']} {hp['activacion']} {hp['optimizador']} "
            f"drop={hp['dropout']} lr={hp['tasa_aprendizaje']:.1e} ({res['segundos']:.1f} s)"
        )
    total = float(time.perf_counter() - t0)

    asc = metrica in {"log_loss", "brier"}
    tabla_ord = pd.DataFrame(filas).sort_values(f"val_{metrica}", ascending=asc).reset_index(drop=True)
    mejor = tabla_ord.iloc[0]

    tabla_csv = tabla_ord.copy()
    tabla_csv["capas_ocultas"] = tabla_csv["capas_ocultas"].astype(str)
    tabla_csv.to_csv(resultados / "busqueda.csv", index=False)
    figura_busqueda(tabla_csv, metrica, figuras / "busqueda_hiperparametros.png")

    mejor_hp = {
        k: (
            [int(x) for x in mejor[k]]
            if k == "capas_ocultas"
            else (
                float(mejor[k])
                if isinstance(mejor[k], (float, np.floating))
                else (int(mejor[k]) if isinstance(mejor[k], (int, np.integer)) else mejor[k])
            )
        )
        for k in CLAVES_HP
    }
    with open(resultados / "mejor_configuracion.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                **mejor_hp,
                "epocas_max": epocas,
                "paciencia": int(cfg["paciencia"]),
                "elegida_por": f"val_{metrica}",
                "configuracion_numero": int(mejor["configuracion"]),
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    resumen = {
        "configuraciones_entrenadas": n_conf,
        "segundos_totales": total,
        "metrica_seleccion": f"val_{metrica}",
        "mejor_valor_validacion": float(mejor[f"val_{metrica}"]),
        "parametros_mejor": int(mejor["parametros"]),
        "mejor": mejor_hp,
    }
    with open(resultados / "busqueda_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    return tabla_csv, resumen


def modo_semillas(config, datos, entradas, val_meta, prueba_meta, resultados, figuras, demo) -> tuple[pd.DataFrame, str]:
    cfg = config["red_neuronal"]["semillas"]
    ruta_conf = Path(config["red_neuronal"]["resultados"]) / "demo" / "mejor_configuracion.yaml" if demo else Path(cfg["configuracion"])
    if ruta_conf.exists():
        with open(ruta_conf, encoding="utf-8") as f:
            base = yaml.safe_load(f)
        origen = str(ruta_conf)
    else:
        base = config["red_neuronal"]["entrenar"]
        origen = "red_neuronal.entrenar (sin archivo de búsqueda)"

    hp = {k: base[k] for k in CLAVES_HP}
    epocas = int(config["red_neuronal"]["demo"]["epocas_max"] if demo else base["epocas_max"])
    n = int(config["red_neuronal"]["demo"]["n_semillas"] if demo else cfg["n"])
    semillas = list(range(int(cfg["primera"]), int(cfg["primera"]) + n))
    metrica = config["red_neuronal"]["metrica_seleccion"]
    maximizar = metrica == "auc"

    filas = []
    mejor = None
    t0 = time.perf_counter()

    for s in semillas:
        res = entrenar_modelo(hp, datos, s, epocas, int(base["paciencia"]))
        ev = evaluar(res["modelo"], datos)
        filas += _filas_metricas(
            ev,
            {
                "semilla": s,
                "mejor_epoca": res["mejor_epoca"],
                "epocas_corridas": res["epocas_corridas"],
                "segundos": res["segundos"],
            },
        )
        print(
            f"  semilla {s:>2}: val_auc={ev['validacion']['auc']:.4f} "
            f"prueba_auc={ev['prueba']['auc']:.4f} val_brier={ev['validacion']['brier']:.4f} ({res['segundos']:.1f} s)"
        )
        score_val = ev["validacion"][metrica]
        es_mejor = mejor is None or (score_val > mejor[0] if maximizar else score_val < mejor[0])
        if es_mejor:
            mejor = (score_val, s, res)

    total = float(time.perf_counter() - t0)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(resultados / "semillas.csv", index=False)

    metricas_cols = ["auc", "gini", "ks", "brier", "log_loss"]
    resumen = tabla.groupby("conjunto")[metricas_cols].agg(["mean", "std", "min", "max"])
    resumen.columns = [f"{m}_{e}" for m, e in resumen.columns]
    resumen = resumen.reindex(CONJUNTOS)
    resumen["semillas"] = n
    resumen["parametros"] = contar_parametros(mejor[2]["modelo"])
    resumen.to_csv(resultados / "resumen_semillas.csv")

    with open(resultados / "semillas_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "configuracion_usada": origen,
                "hiperparametros": hp,
                "semillas": semillas,
                "segundos_totales": total,
                "metrica_seleccion": f"validacion.{metrica}",
                "semilla_del_modelo_guardado": int(mejor[1]),
                "parametros": contar_parametros(mejor[2]["modelo"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    guardar_salidas_modelo(mejor[2], hp, entradas, int(mejor[1]), datos, val_meta, prueba_meta, resultados, figuras)

    if not demo:
        ruta_intercambio = Path("../intercambio/pd_prueba.csv")
        if ruta_intercambio.parent.exists():
            pred_pru = predecir(mejor[2]["modelo"], datos["prueba"][0])
            pd.DataFrame({
                "id": prueba_meta["id"].to_numpy(),
                "pd": np.round(pred_pru, 6),
                "modelo": "mlp_final",
            }).to_csv(ruta_intercambio, index=False)
            print(f"Archivo de intercambio actualizado: {ruta_intercambio} (modelo=mlp_final, {len(pred_pru):,} filas)")

    return resumen, origen


def main() -> None:
    parser = argparse.ArgumentParser(description="Red neuronal (MLP en PyTorch) para PD")
    parser.add_argument("--config", default="configs/pd.yaml")
    parser.add_argument("--modo", choices=["entrenar", "busqueda", "semillas"], default="entrenar")
    parser.add_argument("--demo", action="store_true", help="Corrida corta en subcarpeta demo")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    args = parser.parse_args()

    config = cargar_config(args.config, args.set)
    resultados = Path(config["red_neuronal"]["resultados"])
    figuras = Path(config["red_neuronal"]["figuras"])
    if args.demo:
        resultados, figuras = resultados / "demo", figuras / "demo"
    if args.modo == "entrenar":
        resultados, figuras = resultados / "entrenar", figuras / "entrenar"
    resultados.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    datos, entradas, val_meta, prueba_meta = preparar_datos(config, args.demo)
    pd.set_option("display.width", 140)
    fmt = lambda v: f"{v:.4f}"

    print(
        f"=== RED NEURONAL MLP (PD) | Modo: {args.modo}{' [DEMO]' if args.demo else ''} ===\n"
        f"Entrenamiento: {len(datos['entrenamiento'][1]):,} créditos | Validación: {len(datos['validacion'][1]):,} | "
        f"Prueba: {len(datos['prueba'][1]):,} | Entradas: {len(entradas)}"
    )

    if args.modo == "entrenar":
        tabla = modo_entrenar(config, datos, entradas, val_meta, prueba_meta, resultados, figuras, args.demo)
        print(tabla[["conjunto", "auc", "gini", "ks", "brier", "log_loss", "parametros", "mejor_epoca", "segundos"]]
              .to_string(index=False, float_format=fmt))
    elif args.modo == "busqueda":
        tabla, resumen = modo_busqueda(config, datos, resultados, figuras, args.demo)
        print(
            f"\nBúsqueda finalizada: {resumen['configuraciones_entrenadas']} configuraciones en "
            f"{resumen['segundos_totales']:.1f} s. Top 5 por {resumen['metrica_seleccion']}:"
        )
        print(
            tabla.head(5)[["configuracion", "capas_ocultas", "activacion", "optimizador", "dropout",
                           "tasa_aprendizaje", "val_auc", "val_brier", "parametros"]]
            .to_string(index=False, float_format=fmt)
        )
    else:
        resumen, origen = modo_semillas(config, datos, entradas, val_meta, prueba_meta, resultados, figuras, args.demo)
        print(f"\nResumen de semillas (config: {origen}):")
        cols_res = ["auc_mean", "auc_std", "auc_min", "auc_max", "brier_mean", "ks_mean", "semillas", "parametros"]
        print(resumen[cols_res].to_string(float_format=fmt))


if __name__ == "__main__":
    main()
