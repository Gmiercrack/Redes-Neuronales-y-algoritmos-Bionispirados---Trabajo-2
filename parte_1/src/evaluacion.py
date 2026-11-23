"""
Evaluación de discriminación y calibración para Probabilidad de Incumplimiento (PD).

Compara las predicciones guardadas de la Regresión Logística y la Red Neuronal
(MLP final) sobre validación (2013) y prueba (2014), sin reentrenar modelos:
  - Discriminación: Curva ROC, AUC, Índice de Gini y Estadístico KS.
  - Calibración: Puntaje de Brier, Log-Loss, ECE y curvas de calibración por deciles.

Uso (desde parte_1/):
    python -m src.evaluacion --config configs/pd.yaml
    python -m src.evaluacion --config configs/pd.yaml --demo
    python -m src.evaluacion --config configs/pd.yaml --set evaluacion.grupos_calibracion=20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.preprocesamiento import cargar_config
from src.red_neuronal import AZUL, FONDO, GUIA, NARANJA, TEXTO, TEXTO_2, _estilo
from src.regresion_logistica import calcular_metricas

CONJUNTOS_EVAL = ["validacion", "prueba"]


def cargar_predicciones(config: dict, demo: bool) -> dict[str, pd.DataFrame]:
    """Carga y une por id las predicciones de Regresión Logística y MLP."""
    ruta_lr = Path(config["regresion_logistica"]["resultados"])
    ruta_mlp = Path(config["red_neuronal"]["resultados"])
    if demo:
        ruta_lr = ruta_lr / "demo"
        ruta_mlp = ruta_mlp / "demo"

    datos = {}
    for c in CONJUNTOS_EVAL:
        df_lr = pd.read_parquet(ruta_lr / f"predicciones_{c}.parquet")
        df_mlp = pd.read_parquet(ruta_mlp / f"predicciones_{c}.parquet")
        unido = pd.merge(df_lr, df_mlp, on=["id", "malo"], how="inner")
        if len(unido) != len(df_lr) or len(unido) != len(df_mlp):
            raise ValueError(
                f"Desajuste de filas en '{c}': LR={len(df_lr)}, MLP={len(df_mlp)}, unión={len(unido)}"
            )
        datos[c] = unido
    return datos


def calibracion_por_deciles(y: np.ndarray, pred: np.ndarray, grupos: int = 10) -> pd.DataFrame:
    """Agrupa las predicciones en cuantiles de igual tamaño y compara PD predicha vs observada."""
    tabla = pd.DataFrame({"y": np.asarray(y, dtype=int), "pred": np.asarray(pred, dtype=float)})
    tabla["grupo"] = pd.qcut(tabla["pred"].rank(method="first"), grupos, labels=False) + 1
    cal = (
        tabla.groupby("grupo")
        .agg(
            creditos=("y", "size"),
            malos=("y", "sum"),
            pd_predicha_media=("pred", "mean"),
            tasa_observada=("y", "mean"),
        )
        .reset_index()
    )
    cal["brecha_pp"] = (cal["tasa_observada"] - cal["pd_predicha_media"]) * 100.0
    return cal


def error_calibracion_esperado(cal: pd.DataFrame) -> float:
    """Error de calibración esperado (ECE) ponderado por tamaño de grupo."""
    pesos = cal["creditos"].to_numpy(dtype=float)
    dif = np.abs(cal["tasa_observada"].to_numpy() - cal["pd_predicha_media"].to_numpy())
    return float(np.sum(pesos * dif) / np.sum(pesos))


def curva_ks(y: np.ndarray, pred: np.ndarray) -> dict:
    """Calcula las distribuciones acumuladas de buenos y malos y el punto de máximo KS."""
    fpr, tpr, umbrales = roc_curve(y, pred)
    mask = np.isfinite(umbrales)
    fpr_m, tpr_m, umb_m = fpr[mask], tpr[mask], umbrales[mask]
    diff = np.abs(tpr_m - fpr_m)
    idx_max = int(np.argmax(diff))

    orden = np.argsort(umb_m)
    umb_ord = umb_m[orden]
    f_buenos = 1.0 - fpr_m[orden]
    f_malos = 1.0 - tpr_m[orden]

    p_star = float(umb_m[idx_max])
    return {
        "ks": float(diff[idx_max]),
        "umbral_ks": p_star,
        "f_buenos_ks": float(1.0 - fpr_m[idx_max]),
        "f_malos_ks": float(1.0 - tpr_m[idx_max]),
        "umbrales": umb_ord,
        "f_buenos": f_buenos,
        "f_malos": f_malos,
    }


# =============================================================================
# Figuras
# =============================================================================
def figura_roc(df_prueba: pd.DataFrame, m_lr: dict, m_mlp: dict, ruta: Path) -> None:
    y = df_prueba["malo"].to_numpy(dtype=int)
    fpr_lr, tpr_lr, _ = roc_curve(y, df_prueba["pd_logistica"].to_numpy())
    fpr_mlp, tpr_mlp, _ = roc_curve(y, df_prueba["pd_red"].to_numpy())

    fig, ax = plt.subplots(figsize=(6.8, 5.4), facecolor=FONDO)
    _estilo(ax)
    ax.plot([0, 1], [0, 1], color=GUIA, linestyle=":", linewidth=1.4, label="Azar (AUC = 0.5000)")
    ax.plot(
        fpr_lr,
        tpr_lr,
        color=AZUL,
        linestyle="--",
        linewidth=2.0,
        label=f"Regresión logística (AUC = {m_lr['auc']:.4f}, Gini = {m_lr['gini']:.4f})",
    )
    ax.plot(
        fpr_mlp,
        tpr_mlp,
        color=NARANJA,
        linewidth=2.2,
        label=f"MLP final (AUC = {m_mlp['auc']:.4f}, Gini = {m_mlp['gini']:.4f})",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tasa de falsos positivos (1 - Especificidad)", color=TEXTO, fontsize=10)
    ax.set_ylabel("Tasa de verdaderos positivos (Sensibilidad)", color=TEXTO, fontsize=10)
    ax.set_title("Curva ROC en conjunto de prueba (2014)", color=TEXTO, fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(loc="lower right", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8.8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor=FONDO)
    plt.close(fig)


def figura_ks(df_prueba: pd.DataFrame, ks_mlp: dict, ks_lr: dict, ruta: Path) -> None:
    p_lim = float(min(0.75, max(0.50, np.quantile(df_prueba["pd_red"].to_numpy(), 0.995))))

    fig, ax = plt.subplots(figsize=(7.5, 4.9), facecolor=FONDO)
    _estilo(ax)
    ax.plot(ks_mlp["umbrales"], ks_mlp["f_buenos"], color=AZUL, linewidth=2.2, label="Buenos pagadores acumulados $F_{\\text{buenos}}(p)$")
    ax.plot(ks_mlp["umbrales"], ks_mlp["f_malos"], color=NARANJA, linewidth=2.2, label="Incumplidos acumulados $F_{\\text{malos}}(p)$")

    p_star = ks_mlp["umbral_ks"]
    fb_star, fm_star = ks_mlp["f_buenos_ks"], ks_mlp["f_malos_ks"]
    ax.vlines(p_star, fm_star, fb_star, color=TEXTO, linestyle="--", linewidth=1.8, label=f"Separación máxima KS = {ks_mlp['ks']*100:.2f}% (p* = {p_star:.3f})")
    ax.scatter([p_star, p_star], [fb_star, fm_star], color=TEXTO, s=36, zorder=5)
    ax.annotate(
        f"KS = {ks_mlp['ks']*100:.2f}%\nen PD = {p_star:.3f}",
        xy=(p_star, (fb_star + fm_star) / 2),
        xytext=(min(p_lim * 0.72, p_star + 0.08), (fb_star + fm_star) / 2 - 0.06),
        arrowprops=dict(arrowstyle="->", color=TEXTO_2, lw=1.0),
        fontsize=9,
        color=TEXTO,
        bbox=dict(boxstyle="round,pad=0.3", fc=FONDO, ec=GUIA),
    )

    ax.set_xlim(0, p_lim)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Probabilidad de incumplimiento predicha (PD)", color=TEXTO, fontsize=10)
    ax.set_ylabel("Fracción acumulada de créditos", color=TEXTO, fontsize=10)
    ax.set_title(
        f"Estadístico KS en prueba (2014): MLP = {ks_mlp['ks']*100:.2f}% vs. Reg. Logística = {ks_lr['ks']*100:.2f}%",
        color=TEXTO,
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    ax.legend(loc="lower right", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(ruta, dpi=160, facecolor=FONDO)
    plt.close(fig)


def figura_calibracion(cal_dict: dict, metricas_dict: dict, ruta: Path) -> None:
    titulos = {"validacion": "A. Validación (2013)", "prueba": "B. Prueba (2014)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), facecolor=FONDO, sharey=True)

    max_val = 0.0
    for c in CONJUNTOS_EVAL:
        for m in ["regresion_logistica", "red_neuronal"]:
            max_val = max(
                max_val,
                float(cal_dict[c][m]["pd_predicha_media"].max()),
                float(cal_dict[c][m]["tasa_observada"].max()),
            )
    lim_sup = min(1.0, max(0.50, max_val * 1.08))

    for ax, c in zip(axes, CONJUNTOS_EVAL):
        _estilo(ax, grid_axis="both")
        ax.plot([0, lim_sup], [0, lim_sup], color=GUIA, linestyle=":", linewidth=1.6, label="Calibración perfecta (y = x)")

        c_lr = cal_dict[c]["regresion_logistica"]
        c_mlp = cal_dict[c]["red_neuronal"]
        b_lr = metricas_dict[c]["regresion_logistica"]["brier"]
        b_mlp = metricas_dict[c]["red_neuronal"]["brier"]

        ax.plot(
            c_lr["pd_predicha_media"],
            c_lr["tasa_observada"],
            color=AZUL,
            marker="o",
            linewidth=2.0,
            label=f"Reg. Logística (Brier = {b_lr:.4f})",
        )
        ax.plot(
            c_mlp["pd_predicha_media"],
            c_mlp["tasa_observada"],
            color=NARANJA,
            marker="s",
            linewidth=2.0,
            label=f"MLP final (Brier = {b_mlp:.4f})",
        )

        obs_m = metricas_dict[c]["red_neuronal"]["tasa_malos_obs"]
        pred_m = metricas_dict[c]["red_neuronal"]["pd_media_pred"]
        ax.text(
            0.03,
            lim_sup * 0.90,
            f"Tasa obs.: {obs_m*100:.2f}%\nPD media MLP: {pred_m*100:.2f}%",
            fontsize=8.5,
            color=TEXTO_2,
            bbox=dict(boxstyle="round,pad=0.3", fc=FONDO, ec=GUIA),
        )
        ax.set_xlim(0, lim_sup)
        ax.set_ylim(0, lim_sup)
        ax.set_xlabel("PD media predicha por decil", color=TEXTO, fontsize=9.5)
        ax.set_title(titulos[c], color=TEXTO, fontsize=11, fontweight="bold", loc="left")
        ax.legend(loc="lower right", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8.3)

    axes[0].set_ylabel("Tasa de incumplimiento observada", color=TEXTO, fontsize=9.5)
    fig.suptitle(
        "Curvas de calibración por deciles (el desplazamiento vertical en 2013–2014 refleja la censura temporal)",
        color=TEXTO,
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ruta, dpi=160, facecolor=FONDO)
    plt.close(fig)


# =============================================================================
# Principal
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de discriminación y calibración (PD)")
    parser.add_argument("--config", default="configs/pd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    parser.add_argument("--demo", action="store_true", help="Evalúa los resultados guardados en subcarpetas demo")
    args = parser.parse_args()

    config = cargar_config(args.config, args.set)
    grupos = int(config["evaluacion"]["grupos_calibracion"])
    out_res = Path(config["evaluacion"]["resultados"])
    out_fig = Path(config["evaluacion"]["figuras"])
    if args.demo:
        out_res = out_res / "demo"
        out_fig = out_fig / "demo"
    out_res.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    datos = cargar_predicciones(config, args.demo)
    filas_metricas = []
    cal_dict: dict[str, dict[str, pd.DataFrame]] = {}
    metricas_dict: dict[str, dict[str, dict]] = {}
    ks_dict: dict[str, dict[str, dict]] = {}

    for c in CONJUNTOS_EVAL:
        df_c = datos[c]
        y = df_c["malo"].to_numpy(dtype=int)
        cal_dict[c] = {}
        metricas_dict[c] = {}
        ks_dict[c] = {}

        for mod, col in [("regresion_logistica", "pd_logistica"), ("red_neuronal", "pd_red")]:
            pred = df_c[col].to_numpy(dtype=float)
            m = calcular_metricas(y, pred)
            cal = calibracion_por_deciles(y, pred, grupos=grupos)
            ece = error_calibracion_esperado(cal)
            ks_info = curva_ks(y, pred)

            cal.to_csv(out_res / f"calibracion_{mod}_{c}.csv", index=False)
            cal_dict[c][mod] = cal
            ks_dict[c][mod] = ks_info

            registro = {**m, "umbral_ks": ks_info["umbral_ks"], "ece": ece}
            metricas_dict[c][mod] = registro
            filas_metricas.append({"modelo": mod, "conjunto": c, **registro})

    tabla_metricas = pd.DataFrame(filas_metricas)
    tabla_metricas.to_csv(out_res / "metricas_evaluacion.csv", index=False)

    resumen = {
        "grupos_calibracion": grupos,
        "demo": bool(args.demo),
        "metricas": metricas_dict,
    }
    with open(out_res / "resumen_evaluacion.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    figura_roc(datos["prueba"], metricas_dict["prueba"]["regresion_logistica"], metricas_dict["prueba"]["red_neuronal"], out_fig / "roc_prueba.png")
    figura_ks(datos["prueba"], ks_dict["prueba"]["red_neuronal"], ks_dict["prueba"]["regresion_logistica"], out_fig / "ks_prueba.png")
    figura_calibracion(cal_dict, metricas_dict, out_fig / "calibracion_pd.png")

    pd.set_option("display.width", 160)
    print(f"=== EVALUACIÓN PD: DISCRIMINACIÓN Y CALIBRACIÓN{' [DEMO]' if args.demo else ''} ===")
    cols_print = ["modelo", "conjunto", "auc", "gini", "ks", "umbral_ks", "brier", "log_loss", "ece", "pd_media_pred", "tasa_malos_obs"]
    print(tabla_metricas[cols_print].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nTablas guardadas en: {out_res}")
    print(f"Figuras guardadas en: {out_fig} (roc_prueba.png, ks_prueba.png, calibracion_pd.png)")


if __name__ == "__main__":
    main()
