"""
Modelo de referencia de baja complejidad para PD: Modelo Nulo y Regresión Logística.

Evalúa una rejilla explícita del parámetro de regularización L2 (C) seleccionando
el ganador exclusivamente sobre validación temporal (2013), registra presupuesto de
búsqueda, tiempo de cómputo y parámetros entrenables (55), y guarda coeficientes y
predicciones para evaluación y scorecard.

Uso (desde parte_1/):
    python -m src.regresion_logistica --config configs/pd.yaml
    python -m src.regresion_logistica --config configs/pd.yaml --demo
    python -m src.regresion_logistica --config configs/pd.yaml --set regresion_logistica.rejilla_c=[0.01,0.1,1.0]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from src.preprocesamiento import cargar_config

CONJUNTOS = ["entrenamiento", "validacion", "prueba"]
PREFIJOS_ENTRADA = ("num__", "cat__")


def calcular_metricas(y_true: np.ndarray | pd.Series, y_prob: np.ndarray) -> dict[str, float]:
    """Calcula métricas de discriminación (AUC, Gini, KS) y calibración (Brier, Log-Loss)."""
    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(y_prob, dtype=float)
    auc = float(roc_auc_score(y_arr, p_arr))
    gini = float(2.0 * auc - 1.0)
    fpr, tpr, _ = roc_curve(y_arr, p_arr)
    ks = float(np.max(np.abs(tpr - fpr)))
    brier = float(brier_score_loss(y_arr, p_arr))
    ll = float(log_loss(y_arr, np.clip(p_arr, 1e-15, 1.0 - 1e-15), labels=[0, 1]))
    return {
        "n": int(len(y_arr)),
        "tasa_malos_obs": float(np.mean(y_arr)),
        "pd_media_pred": float(np.mean(p_arr)),
        "auc": auc,
        "gini": gini,
        "ks": ks,
        "brier": brier,
        "log_loss": ll,
    }


def cargar_conjuntos(config: dict, demo: bool) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Carga los parquet procesados de PD y aplica submuestreo en entrenamiento si --demo."""
    carpeta = Path(config["rutas"]["salida"])
    tablas = {c: pd.read_parquet(carpeta / f"pd_{c}.parquet") for c in CONJUNTOS}
    if demo:
        frac = float(config["regresion_logistica"]["demo"]["fraccion_entrenamiento"])
        tablas["entrenamiento"] = tablas["entrenamiento"].sample(
            frac=frac, random_state=int(config["semilla"])
        ).reset_index(drop=True)
    entradas = [c for c in tablas["entrenamiento"].columns if c.startswith(PREFIJOS_ENTRADA)]
    if not entradas:
        raise ValueError("No se encontraron columnas de entrada con prefijo num__ o cat__.")
    return tablas, entradas


def evaluar_modelo_nulo(tablas: dict[str, pd.DataFrame]) -> tuple[float, list[dict]]:
    """Evalúa el modelo constante que predice siempre la tasa base de entrenamiento."""
    p_base = float(tablas["entrenamiento"]["malo"].astype(int).mean())
    filas = []
    for c in CONJUNTOS:
        y = tablas[c]["malo"].astype(int).to_numpy()
        pred = np.full(len(y), p_base, dtype=float)
        filas.append({"modelo": "modelo_nulo", "conjunto": c, **calcular_metricas(y, pred)})
    return p_base, filas


def ejecutar_rejilla(
    tablas: dict[str, pd.DataFrame],
    entradas: list[str],
    config: dict,
    demo: bool,
) -> tuple[pd.DataFrame, LogisticRegression, float, float]:
    """Entrena un LogisticRegression por cada C de la rejilla y elige el mejor en validación."""
    cfg = config["regresion_logistica"]
    rejilla = cfg["demo"]["rejilla_c"] if demo else cfg["rejilla_c"]
    max_iter = int(cfg["max_iter"])
    metrica_sel = cfg["metrica_seleccion"]
    semilla = int(config["semilla"])

    if metrica_sel not in {"auc", "brier", "log_loss"}:
        raise ValueError(f"metrica_seleccion inválida para regresión logística: {metrica_sel}")

    X_tr = tablas["entrenamiento"][entradas].to_numpy(dtype=float)
    y_tr = tablas["entrenamiento"]["malo"].astype(int).to_numpy()
    X_va = tablas["validacion"][entradas].to_numpy(dtype=float)
    y_va = tablas["validacion"]["malo"].astype(int).to_numpy()

    filas_grid = []
    modelos_ajustados: dict[float, LogisticRegression] = {}
    t_inicio_total = time.perf_counter()

    for idx, c_val in enumerate(rejilla, start=1):
        c_float = float(c_val)
        t0 = time.perf_counter()
        lr = LogisticRegression(
            C=c_float,
            solver="lbfgs",
            max_iter=max_iter,
            random_state=semilla,
        )
        lr.fit(X_tr, y_tr)
        seg = time.perf_counter() - t0

        prob_tr = lr.predict_proba(X_tr)[:, 1]
        prob_va = lr.predict_proba(X_va)[:, 1]
        m_tr = calcular_metricas(y_tr, prob_tr)
        m_va = calcular_metricas(y_va, prob_va)

        modelos_ajustados[c_float] = lr
        filas_grid.append({
            "configuracion": idx,
            "C": c_float,
            "iteraciones": int(lr.n_iter_[0]),
            "segundos": float(seg),
            "entr_auc": m_tr["auc"],
            "entr_brier": m_tr["brier"],
            "entr_log_loss": m_tr["log_loss"],
            "val_auc": m_va["auc"],
            "val_gini": m_va["gini"],
            "val_ks": m_va["ks"],
            "val_brier": m_va["brier"],
            "val_log_loss": m_va["log_loss"],
        })

    segundos_totales = float(time.perf_counter() - t_inicio_total)
    tabla_grid = pd.DataFrame(filas_grid)
    ascendente = metrica_sel in {"brier", "log_loss"}
    tabla_ordenada = tabla_grid.sort_values(f"val_{metrica_sel}", ascending=ascendente).reset_index(drop=True)
    best_c = float(tabla_ordenada.loc[0, "C"])
    return tabla_grid, modelos_ajustados[best_c], best_c, segundos_totales


def guardar_resultados(
    tablas: dict[str, pd.DataFrame],
    entradas: list[str],
    p_base: float,
    filas_nulo: list[dict],
    tabla_grid: pd.DataFrame,
    best_lr: LogisticRegression,
    best_c: float,
    segundos_totales: float,
    config: dict,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Guarda tabla de rejilla, métricas, coeficientes, joblib, predicciones y resumen JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    filas_lr = []
    metricas_por_conjunto = {"modelo_nulo": {}, "regresion_logistica": {}}

    for fila in filas_nulo:
        metricas_por_conjunto["modelo_nulo"][fila["conjunto"]] = {
            k: v for k, v in fila.items() if k not in {"modelo", "conjunto"}
        }

    for c in CONJUNTOS:
        X_c = tablas[c][entradas].to_numpy(dtype=float)
        y_c = tablas[c]["malo"].astype(int).to_numpy()
        pred_c = best_lr.predict_proba(X_c)[:, 1]
        m_c = calcular_metricas(y_c, pred_c)
        filas_lr.append({"modelo": "regresion_logistica", "conjunto": c, **m_c})
        metricas_por_conjunto["regresion_logistica"][c] = m_c

        if c in {"validacion", "prueba"}:
            pd.DataFrame({
                "id": tablas[c]["id"].to_numpy(),
                "malo": y_c,
                "pd_logistica": pred_c,
            }).to_parquet(out_dir / f"predicciones_{c}.parquet", index=False)

    tabla_metricas = pd.DataFrame(filas_nulo + filas_lr)
    tabla_grid.to_csv(out_dir / "rejilla_c.csv", index=False)
    tabla_metricas.to_csv(out_dir / "metricas.csv", index=False)

    coefs = pd.DataFrame({
        "variable": entradas,
        "coeficiente": best_lr.coef_.ravel().astype(float),
    })
    coefs["abs_coeficiente"] = coefs["coeficiente"].abs()
    coefs = coefs.sort_values("abs_coeficiente", ascending=False).reset_index(drop=True)
    coefs.to_csv(out_dir / "coeficientes_logistica.csv", index=False)

    parametros = int(best_lr.coef_.size + best_lr.intercept_.size)
    joblib.dump(
        {
            "modelo": best_lr,
            "entradas": entradas,
            "C": best_c,
            "semilla": int(config["semilla"]),
            "intercepto": float(best_lr.intercept_[0]),
        },
        out_dir / "modelo_logistica.joblib",
    )

    resumen = {
        "configuraciones_evaluadas": int(len(tabla_grid)),
        "mejor_C": best_c,
        "metrica_seleccion": config["regresion_logistica"]["metrica_seleccion"],
        "parametros_entrenables": parametros,
        "intercepto": float(best_lr.intercept_[0]),
        "segundos_totales_rejilla": segundos_totales,
        "tasa_base_entrenamiento": p_base,
        "metricas": metricas_por_conjunto,
    }
    with open(out_dir / "resumen_regresion_logistica.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    return tabla_metricas, resumen


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelo de referencia: Regresión Logística (PD)")
    parser.add_argument("--config", default="configs/pd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    parser.add_argument("--demo", action="store_true", help="Corrida corta en subcarpeta demo")
    args = parser.parse_args()

    config = cargar_config(args.config, args.set)
    out_dir = Path(config["regresion_logistica"]["resultados"])
    if args.demo:
        out_dir = out_dir / "demo"

    tablas, entradas = cargar_conjuntos(config, args.demo)
    p_base, filas_nulo = evaluar_modelo_nulo(tablas)
    tabla_grid, best_lr, best_c, seg_tot = ejecutar_rejilla(tablas, entradas, config, args.demo)
    tabla_metricas, resumen = guardar_resultados(
        tablas, entradas, p_base, filas_nulo, tabla_grid, best_lr, best_c, seg_tot, config, out_dir
    )

    pd.set_option("display.width", 140)
    print(f"=== REGRESIÓN LOGÍSTICA (PD){' [DEMO]' if args.demo else ''} ===")
    print(f"Entradas: {len(entradas)} | Parámetros entrenables: {resumen['parametros_entrenables']} | "
          f"Presupuesto rejilla: {resumen['configuraciones_evaluadas']} configs en {seg_tot:.2f} s")
    print("\nBúsqueda en rejilla de C (selección por validación):")
    print(tabla_grid[["configuracion", "C", "iteraciones", "segundos", "val_auc", "val_brier", "val_log_loss"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nMejor C (en validación): {best_c} | Salida: {out_dir}")
    print("\nMétricas finales (Modelo Nulo vs Regresión Logística):")
    print(tabla_metricas[["modelo", "conjunto", "n", "auc", "gini", "ks", "brier", "log_loss"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
