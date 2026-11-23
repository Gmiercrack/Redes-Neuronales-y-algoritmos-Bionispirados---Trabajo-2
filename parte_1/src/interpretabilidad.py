"""
Interpretabilidad mediante Importancia por Permutación (Permutation Feature Importance).

Evalúa la importancia de las 21 variables originales en el conjunto de prueba (2014)
para el MLP final y la Regresión Logística (Breiman, 2001; Fisher et al., 2019),
agrupando por variable original las columnas derivadas (indicador de faltante y
codificación one-hot) para permutar perfiles completos y coherentes.

Uso (desde parte_1/):
    python -m src.interpretabilidad --config configs/pd.yaml
    python -m src.interpretabilidad --config configs/pd.yaml --demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.preprocesamiento import cargar_config
from src.red_neuronal import AZUL, FONDO, GUIA, MLP, NARANJA, TEXTO, TEXTO_2, _estilo, predecir


def agrupar_columnas_originales(config: dict, entradas: list[str]) -> dict[str, list[int]]:
    """Mapea cada una de las 21 variables originales a sus índices en la matriz transformada."""
    idx_map = {col: i for i, col in enumerate(entradas)}
    grupos: dict[str, list[int]] = {}

    for var in config["variables"]["numericas"]:
        cols = [c for c in (f"num__{var}", f"num__missingindicator_{var}") if c in idx_map]
        if not cols:
            raise ValueError(f"No se encontraron columnas transformadas para '{var}'")
        grupos[var] = [idx_map[c] for c in cols]

    for var in config["variables"]["categoricas"]:
        prefijo = f"cat__{var}_"
        cols = [c for c in entradas if c.startswith(prefijo)]
        if not cols:
            raise ValueError(f"No se encontraron columnas one-hot para '{var}'")
        grupos[var] = [idx_map[c] for c in cols]

    cubiertas = [idx for idxs in grupos.values() for idx in idxs]
    if len(cubiertas) != len(entradas) or len(set(cubiertas)) != len(entradas):
        raise ValueError(f"Cobertura inconsistente de columnas: {len(cubiertas)} vs {len(entradas)}")
    return grupos


def calcular_importancia_permutacion(
    X_test: np.ndarray,
    y_test: np.ndarray,
    grupos: dict[str, list[int]],
    modelo_mlp: MLP,
    modelo_lr,
    n_rep: int,
    semilla: int,
) -> tuple[pd.DataFrame, float, float]:
    """Calcula la caída media y desviación estándar del AUC al permutar cada variable."""
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    auc_base_mlp = float(roc_auc_score(y_test, predecir(modelo_mlp, X_tensor)))
    auc_base_lr = float(roc_auc_score(y_test, modelo_lr.predict_proba(X_test)[:, 1]))

    rng = np.random.default_rng(semilla)
    n_filas = len(y_test)
    filas = []

    for var, idxs in grupos.items():
        caidas_mlp, caidas_lr = [], []
        X_perm = X_test.copy()
        for _ in range(n_rep):
            perm = rng.permutation(n_filas)
            X_perm[:, idxs] = X_test[perm][:, idxs]

            pred_mlp = predecir(modelo_mlp, torch.tensor(X_perm, dtype=torch.float32))
            pred_lr = modelo_lr.predict_proba(X_perm)[:, 1]

            caidas_mlp.append(auc_base_mlp - float(roc_auc_score(y_test, pred_mlp)))
            caidas_lr.append(auc_base_lr - float(roc_auc_score(y_test, pred_lr)))

        # Restaurar las columnas originales antes de pasar a la siguiente variable
        X_perm[:, idxs] = X_test[:, idxs]
        filas.append(
            {
                "variable": var,
                "n_columnas": len(idxs),
                "mlp_delta_auc_mean": float(np.mean(caidas_mlp)),
                "mlp_delta_auc_std": float(np.std(caidas_mlp)),
                "lr_delta_auc_mean": float(np.mean(caidas_lr)),
                "lr_delta_auc_std": float(np.std(caidas_lr)),
            }
        )

    tabla = pd.DataFrame(filas).sort_values("mlp_delta_auc_mean", ascending=False).reset_index(drop=True)
    return tabla, auc_base_mlp, auc_base_lr


def figura_importancia(tabla: pd.DataFrame, ruta: Path, top_k: int = 15) -> None:
    top = tabla.head(min(top_k, len(tabla))).iloc[::-1]
    y = np.arange(len(top))
    h = 0.36

    fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor=FONDO)
    _estilo(ax, grid_axis="x")

    ax.barh(
        y + h / 2,
        top["mlp_delta_auc_mean"] * 100.0,
        height=h,
        xerr=top["mlp_delta_auc_std"] * 100.0,
        color=NARANJA,
        capsize=2.5,
        label="MLP final",
    )
    ax.barh(
        y - h / 2,
        top["lr_delta_auc_mean"] * 100.0,
        height=h,
        xerr=top["lr_delta_auc_std"] * 100.0,
        color=AZUL,
        capsize=2.5,
        label="Regresión logística",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(top["variable"], fontsize=9, color=TEXTO)
    ax.set_xlabel("Caída media del AUC en prueba al permutar (puntos de AUC × 100)", color=TEXTO, fontsize=9.5)
    ax.set_title(
        "Importancia por permutación agrupada (Top 15 variables en prueba 2014)",
        color=TEXTO,
        fontsize=11.5,
        fontweight="bold",
        loc="left",
    )
    ax.legend(loc="lower right", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor=FONDO)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpretabilidad por permutación (PD)")
    parser.add_argument("--config", default="configs/pd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    parser.add_argument("--demo", action="store_true", help="Corrida rápida con 3 repeticiones en demo/")
    args = parser.parse_args()

    config = cargar_config(args.config, args.set)
    interp_cfg = config["interpretabilidad"]
    n_rep = 3 if args.demo else int(interp_cfg["repeticiones_permutacion"])
    semilla = int(interp_cfg["semilla"])

    ruta_lr = Path(config["regresion_logistica"]["resultados"])
    ruta_mlp = Path(config["red_neuronal"]["resultados"])
    out_res = Path(interp_cfg["resultados"])
    out_fig = Path(interp_cfg["figuras"])
    if args.demo:
        ruta_lr, ruta_mlp = ruta_lr / "demo", ruta_mlp / "demo"
        out_res, out_fig = out_res / "demo", out_fig / "demo"
    out_res.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    paq_mlp = torch.load(ruta_mlp / "modelo_red.pt", weights_only=False)
    entradas = paq_mlp["entradas"]
    hp = paq_mlp["hiperparametros"]
    modelo_mlp = MLP(len(entradas), hp["capas_ocultas"], hp["activacion"], hp["dropout"])
    modelo_mlp.load_state_dict(paq_mlp["estado"])
    modelo_mlp.eval()

    paq_lr = joblib.load(ruta_lr / "modelo_logistica.joblib")
    modelo_lr = paq_lr["modelo"]

    df_test = pd.read_parquet(Path(config["rutas"]["salida"]) / "pd_prueba.parquet")
    X_test = df_test[entradas].to_numpy(dtype=float)
    y_test = df_test["malo"].astype(int).to_numpy()
    grupos = agrupar_columnas_originales(config, entradas)

    tabla, auc_mlp, auc_lr = calcular_importancia_permutacion(
        X_test, y_test, grupos, modelo_mlp, modelo_lr, n_rep, semilla
    )
    tabla.to_csv(out_res / "importancia_permutacion.csv", index=False)
    figura_importancia(tabla, out_fig / "importancia_permutacion.png", top_k=15)

    resumen = {
        "repeticiones": n_rep,
        "semilla": semilla,
        "auc_base_mlp": auc_mlp,
        "auc_base_logistica": auc_lr,
        "top_10_variables": tabla.head(10).to_dict(orient="records"),
    }
    with open(out_res / "resumen_interpretabilidad.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    pd.set_option("display.width", 140)
    print(f"=== INTERPRETABILIDAD POR PERMUTACIÓN (PD){' [DEMO]' if args.demo else ''} ===")
    print(f"AUC base prueba -> MLP: {auc_mlp:.4f} | Reg. Logística: {auc_lr:.4f} | Repeticiones: {n_rep}")
    print("\nTop 10 variables por caída de AUC (MLP vs Regresión Logística):")
    print(tabla.head(10).to_string(index=False, float_format=lambda v: f"{v:.5f}"))


if __name__ == "__main__":
    main()
