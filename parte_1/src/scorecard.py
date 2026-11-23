"""
Construcción y evaluación de la Scorecard crediticia para Probabilidad de Incumplimiento (PD).

Transforma la probabilidad de incumplimiento p en puntaje crediticio usando:
    Puntaje(p) = P0 + (PDO / ln(2)) * [ ln((1 - p) / p) - ln(theta0) ]

Incluye:
  - Descomposición aditiva exacta por variable para la Regresión Logística.
  - Verificación numérica de consistencia entre suma de puntos y fórmula log-odds.
  - Bandas decílicas, percentiles poblacionales y figura de distribución en prueba (2014).

Uso (desde parte_1/):
    python -m src.scorecard --config configs/pd.yaml
    python -m src.scorecard --config configs/pd.yaml --demo
    python -m src.scorecard --config configs/pd.yaml --set scorecard.pdo=40 --set scorecard.puntaje_base=700
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

from src.preprocesamiento import cargar_config
from src.red_neuronal import AZUL, FONDO, GUIA, NARANJA, TEXTO, TEXTO_2, _estilo


def prob_a_puntaje(p: np.ndarray, a: float, b: float) -> np.ndarray:
    """Convierte probabilidades de incumplimiento en puntaje de scorecard."""
    p_clip = np.clip(np.asarray(p, dtype=float), 1e-15, 1.0 - 1e-15)
    return a + b * np.log((1.0 - p_clip) / p_clip)


def construir_tabla_puntos(
    modelo_lr,
    entradas: list[str],
    a: float,
    b: float,
) -> tuple[pd.DataFrame, float]:
    """Descompone la Scorecard aditiva exacta a partir de los pesos de la Regresión Logística."""
    beta_0 = float(modelo_lr.intercept_[0])
    betas = modelo_lr.coef_.ravel().astype(float)
    puntaje_base_aditivo = float(a - b * beta_0)

    filas = [
        {
            "variable": "INTERCEPTO_BASE",
            "tipo": "constante_base",
            "coeficiente_beta": beta_0,
            "puntos_por_unidad": puntaje_base_aditivo,
            "abs_puntos": abs(puntaje_base_aditivo),
        }
    ]
    for col, beta_j in zip(entradas, betas):
        if col.startswith("num__missingindicator_"):
            tipo = "indicador_faltante (1 si falta)"
        elif col.startswith("num__"):
            tipo = "numerica (+1 desv. est.)"
        else:
            tipo = "categorica (1 si activa)"
        pts = float(-b * beta_j)
        filas.append(
            {
                "variable": col,
                "tipo": tipo,
                "coeficiente_beta": float(beta_j),
                "puntos_por_unidad": pts,
                "abs_puntos": abs(pts),
            }
        )

    df_pts = pd.DataFrame(filas)
    cuerpo = df_pts.iloc[1:].sort_values("abs_puntos", ascending=False)
    return pd.concat([df_pts.iloc[:1], cuerpo], ignore_index=True), puntaje_base_aditivo


def resumen_por_bandas(df: pd.DataFrame, grupos: int = 10) -> pd.DataFrame:
    """Agrupa en deciles de puntaje del MLP (de menor a mayor puntaje)."""
    tmp = df.copy()
    tmp["banda"] = pd.qcut(tmp["puntaje_red"].rank(method="first"), grupos, labels=False) + 1
    resumen = (
        tmp.groupby("banda")
        .agg(
            creditos=("malo", "size"),
            malos=("malo", "sum"),
            puntaje_min=("puntaje_red", "min"),
            puntaje_max=("puntaje_red", "max"),
            puntaje_medio=("puntaje_red", "mean"),
            pd_media_predicha=("pd_red", "mean"),
            tasa_malos_observada=("malo", "mean"),
        )
        .reset_index()
    )
    return resumen


def figura_distribucion_puntajes(
    df_prueba: pd.DataFrame,
    bandas: pd.DataFrame,
    p0: float,
    pdo: float,
    theta0: float,
    ruta: Path,
) -> None:
    buenos = df_prueba.loc[df_prueba["malo"] == 0, "puntaje_red"].to_numpy()
    malos = df_prueba.loc[df_prueba["malo"] == 1, "puntaje_red"].to_numpy()

    lo = float(np.quantile(df_prueba["puntaje_red"], 0.002))
    hi = float(np.quantile(df_prueba["puntaje_red"], 0.998))
    bins = np.linspace(lo, hi, 45)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), facecolor=FONDO)

    ax0 = axes[0]
    _estilo(ax0)
    ax0.hist(buenos, bins=bins, density=True, alpha=0.55, color=AZUL, label=f"Buenos (0): mediana = {np.median(buenos):.1f}")
    ax0.hist(malos, bins=bins, density=True, alpha=0.55, color=NARANJA, label=f"Malos (1): mediana = {np.median(malos):.1f}")
    ax0.axvline(np.median(buenos), color=AZUL, linestyle="--", linewidth=1.6)
    ax0.axvline(np.median(malos), color=NARANJA, linestyle="--", linewidth=1.6)
    ax0.axvline(p0, color=TEXTO, linestyle=":", linewidth=1.5, label=f"Puntaje base P0 = {p0:.0f} (odds {theta0:.0f}:1)")
    ax0.set_xlabel("Puntaje en la Scorecard (MLP final)", color=TEXTO, fontsize=9.5)
    ax0.set_ylabel("Densidad", color=TEXTO, fontsize=9.5)
    ax0.set_title("A. Distribución de puntajes en prueba (2014)", color=TEXTO, fontsize=10.5, fontweight="bold")
    ax0.legend(loc="upper left", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8.3)

    ax1 = axes[1]
    _estilo(ax1)
    etiquetas = [f"D{int(r['banda'])}\n[{r['puntaje_min']:.0f}–{r['puntaje_max']:.0f}]" for _, r in bandas.iterrows()]
    tasas = bandas["tasa_malos_observada"].to_numpy() * 100.0
    tasa_global = float(df_prueba["malo"].mean() * 100.0)
    colores_b = [NARANJA if t > tasa_global else AZUL for t in tasas]
    bars = ax1.bar(range(len(bandas)), tasas, color=colores_b)
    ax1.axhline(tasa_global, color=TEXTO, linestyle="--", linewidth=1.1, label=f"Tasa promedio ({tasa_global:.1f}%)")
    for b, val in zip(bars, tasas):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{val:.1f}%", ha="center", fontsize=8, color=TEXTO)
    ax1.set_xticks(range(len(bandas)))
    ax1.set_xticklabels(etiquetas, fontsize=7.5)
    ax1.set_ylim(0, max(tasas) * 1.18)
    ax1.set_xlabel("Decil de puntaje (de mayor a menor riesgo)", color=TEXTO, fontsize=9.5)
    ax1.set_ylabel("Tasa de incumplimiento observada (%)", color=TEXTO, fontsize=9.5)
    ax1.set_title("B. Riesgo observado por banda de puntaje", color=TEXTO, fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8.5)

    fig.suptitle(f"Scorecard Crediticia (P0 = {p0:.0f}, θ0 = {theta0:.0f}:1, PDO = {pdo:.0f})",
                 color=TEXTO, fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(ruta, dpi=160, facecolor=FONDO)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Construcción de Scorecard (Parte 1: PD)")
    parser.add_argument("--config", default="configs/pd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    parser.add_argument("--demo", action="store_true", help="Usa resultados de demo y guarda en demo/")
    args = parser.parse_args()

    config = cargar_config(args.config, args.set)
    sc = config["scorecard"]
    p0 = float(sc["puntaje_base"])
    theta0 = float(sc["odds_referencia"])
    pdo = float(sc["pdo"])
    b = float(pdo / np.log(2.0))
    a = float(p0 - b * np.log(theta0))

    ruta_lr = Path(config["regresion_logistica"]["resultados"])
    ruta_mlp = Path(config["red_neuronal"]["resultados"])
    out_res = Path(sc["resultados"])
    out_fig = Path(sc["figuras"])
    if args.demo:
        ruta_lr, ruta_mlp = ruta_lr / "demo", ruta_mlp / "demo"
        out_res, out_fig = out_res / "demo", out_fig / "demo"
    out_res.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    paq_lr = joblib.load(ruta_lr / "modelo_logistica.joblib")
    modelo_lr = paq_lr["modelo"]
    entradas = paq_lr["entradas"]

    df_lr = pd.read_parquet(ruta_lr / "predicciones_prueba.parquet")
    df_mlp = pd.read_parquet(ruta_mlp / "predicciones_prueba.parquet")
    df_prueba = pd.merge(df_lr, df_mlp, on=["id", "malo"], how="inner")
    df_raw_test = pd.read_parquet(Path(config["rutas"]["salida"]) / "pd_prueba.parquet")

    df_prueba["puntaje_logistica"] = prob_a_puntaje(df_prueba["pd_logistica"].to_numpy(), a, b)
    df_prueba["puntaje_red"] = prob_a_puntaje(df_prueba["pd_red"].to_numpy(), a, b)

    tabla_puntos, base_aditiva = construir_tabla_puntos(modelo_lr, entradas, a, b)
    tabla_puntos.to_csv(out_res / "tabla_puntos_variables.csv", index=False)

    # Verificar que la suma aditiva de puntos reproduce el puntaje logístico exacto
    X_test = df_raw_test[entradas].to_numpy(dtype=float)
    pesos_puntos = -b * modelo_lr.coef_.ravel().astype(float)
    puntaje_aditivo = base_aditiva + X_test @ pesos_puntos
    error_max_aditivo = float(np.max(np.abs(puntaje_aditivo - df_prueba["puntaje_logistica"].to_numpy())))

    bandas = resumen_por_bandas(df_prueba, grupos=10)
    bandas.to_csv(out_res / "resumen_bandas_scorecard.csv", index=False)
    df_prueba.to_parquet(out_res / "puntajes_prueba.parquet", index=False)

    percentiles_lista = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
    pct_red = {f"p{p}": float(np.percentile(df_prueba["puntaje_red"], p)) for p in percentiles_lista}
    pct_lr = {f"p{p}": float(np.percentile(df_prueba["puntaje_logistica"], p)) for p in percentiles_lista}

    resumen = {
        "parametros": {"P0": p0, "odds_referencia": theta0, "pdo": pdo, "A": a, "B": b},
        "puntaje_base_aditivo_logistica": base_aditiva,
        "error_max_reconstruccion_aditiva": error_max_aditivo,
        "estadisticas_prueba_mlp": {
            "media_global": float(df_prueba["puntaje_red"].mean()),
            "std_global": float(df_prueba["puntaje_red"].std()),
            "mediana_buenos": float(df_prueba.loc[df_prueba["malo"] == 0, "puntaje_red"].median()),
            "mediana_malos": float(df_prueba.loc[df_prueba["malo"] == 1, "puntaje_red"].median()),
            "percentiles": pct_red,
        },
        "estadisticas_prueba_logistica": {
            "media_global": float(df_prueba["puntaje_logistica"].mean()),
            "mediana_buenos": float(df_prueba.loc[df_prueba["malo"] == 0, "puntaje_logistica"].median()),
            "mediana_malos": float(df_prueba.loc[df_prueba["malo"] == 1, "puntaje_logistica"].median()),
            "percentiles": pct_lr,
        },
    }
    with open(out_res / "resumen_scorecard.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    figura_distribucion_puntajes(df_prueba, bandas, p0, pdo, theta0, out_fig / "distribucion_puntajes.png")

    pd.set_option("display.width", 140)
    print(f"=== SCORECARD CREDITICIA (PD){' [DEMO]' if args.demo else ''} ===")
    print(f"P0 = {p0:.0f} | theta0 = {theta0:.0f}:1 | PDO = {pdo:.0f} | A = {a:.4f} | B = {b:.4f}")
    print(f"Verificación aditiva Reg. Logística -> error máx = {error_max_aditivo:.2e}")
    print(f"MLP prueba -> Mediana buenos: {resumen['estadisticas_prueba_mlp']['mediana_buenos']:.1f} | "
          f"Mediana malos: {resumen['estadisticas_prueba_mlp']['mediana_malos']:.1f}")
    print("\nBandas por decil de puntaje (MLP):")
    print(bandas.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
