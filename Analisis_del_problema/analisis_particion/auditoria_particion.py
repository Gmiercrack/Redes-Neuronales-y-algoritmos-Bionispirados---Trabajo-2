"""
Auditoría empírica de la partición temporal y análisis de censura (2007-2015).

Lee directamente data/raw/loan.csv y reproduce:
  1. Conteos exactos por año de originación y las 10 categorías de loan_status.
  2. Tabla de maduración física por año y plazo (36 vs 60 meses).
  3. Comparación cuantitativa de las 4 estrategias de corte temporal.
  4. Prueba de sensibilidad Opción 1 vs Opción 3 (si existen los parquets procesados).
  5. Figura editorial de censura y tasa de malos por año y plazo.

Uso (desde la raíz del repositorio):
    .\\.venv\\Scripts\\python.exe Analisis_del_problema/analisis_particion/auditoria_particion.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ_REPO = Path(__file__).resolve().parents[2]
DIR_ACTUAL = Path(__file__).resolve().parent

ESTADOS_BUENOS = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
ESTADOS_MALOS = {
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Does not meet the credit policy. Status:Charged Off",
}
ESTADOS_LGD = {
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
}

# Paleta editorial compartida con Parte 2 (referencia_lgd.py)
AZUL, NARANJA = "#2a78d6", "#eb6834"
GRIS_CLARO = "#b8b6b0"
TEXTO, TEXTO_2, FONDO, GUIA = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d4"


def _estilo_ejes(ax) -> None:
    ax.set_facecolor(FONDO)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GUIA)
    ax.spines["bottom"].set_color(GUIA)
    ax.tick_params(colors=TEXTO_2, labelsize=9)
    ax.grid(axis="y", color=GUIA, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def a_fecha_mes(serie: pd.Series) -> pd.Series:
    fechas = pd.to_datetime(serie, format="%b-%Y", errors="coerce")
    if fechas.isna().mean() > 0.5:
        fechas = pd.to_datetime(serie, errors="coerce", format="mixed")
    return fechas


def cargar_crudo_particion(ruta_csv: Path, muestra: float | None = None, semilla: int = 2026) -> pd.DataFrame:
    cols = ["id", "issue_d", "loan_status", "term", "recoveries"]
    df = pd.read_csv(ruta_csv, usecols=cols, low_memory=False)
    if muestra:
        df = df.sample(frac=muestra, random_state=semilla).reset_index(drop=True)
    df["issue_d"] = a_fecha_mes(df["issue_d"])
    df["anio"] = df["issue_d"].dt.year
    df["loan_status"] = df["loan_status"].astype(str).str.strip()
    df["plazo_meses"] = pd.to_numeric(df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["recoveries"] = pd.to_numeric(df["recoveries"], errors="coerce").fillna(0.0)

    df["desenlace"] = "Abierto (NA)"
    df.loc[df["loan_status"].isin(ESTADOS_BUENOS), "desenlace"] = "Bueno (0)"
    df.loc[df["loan_status"].isin(ESTADOS_MALOS), "desenlace"] = "Malo (1)"
    df["castigado"] = df["loan_status"].isin(ESTADOS_LGD)
    return df


def generar_tablas_auditoria(df: pd.DataFrame, out_res: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1. Conteos por año y las 10 categorías de loan_status
    conteos_estado = pd.crosstab(df["anio"], df["loan_status"], margins=True, margins_name="TOTAL")
    conteos_estado.to_csv(out_res / "conteos_por_anio_y_estado.csv")

    # 2. Maduración por año y plazo (36 vs 60 meses)
    mad = (
        df.groupby(["anio", "plazo_meses"], dropna=False)
        .agg(
            buenos=("desenlace", lambda s: int((s == "Bueno (0)").sum())),
            malos=("desenlace", lambda s: int((s == "Malo (1)").sum())),
            abiertos=("desenlace", lambda s: int((s == "Abierto (NA)").sum())),
            total=("id", "size"),
        )
        .reset_index()
    )
    mad["cerrados"] = mad["buenos"] + mad["malos"]
    mad["pct_abierto_censura"] = np.where(mad["total"] > 0, mad["abiertos"] / mad["total"] * 100, 0.0)
    mad["pct_malos_sobre_cerrados"] = np.where(mad["cerrados"] > 0, mad["malos"] / mad["cerrados"] * 100, 0.0)
    mad.to_csv(out_res / "maduracion_por_anio_y_plazo.csv", index=False)

    # 3. Evaluación de las 4 estrategias de corte
    estrategias = {
        "Opcion 1 (Actual: 2012 / 2013 / 2014)": ("2012-12-31", "2013-12-31", "2014-12-31"),
        "Opcion 2 (Alt A: 2011 / 2012 / 2013)": ("2011-12-31", "2012-12-31", "2013-12-31"),
        "Opcion 3 (Alt B: 2013H1 / 2013H2 / 2014H1)": ("2013-06-30", "2013-12-31", "2014-06-30"),
        "Opcion 4 (Alt C: 2013 / 2014H1 / 2014H2)": ("2013-12-31", "2014-06-30", "2014-12-31"),
    }
    filas_est = []
    cerrados_mask = df["desenlace"].isin(["Bueno (0)", "Malo (1)"])
    for nombre, (f_ent, f_val, f_pru) in estrategias.items():
        t_ent, t_val, t_pru = pd.Timestamp(f_ent), pd.Timestamp(f_val), pd.Timestamp(f_pru)
        part = np.select(
            [df["issue_d"] <= t_ent, df["issue_d"] <= t_val, df["issue_d"] <= t_pru, df["issue_d"] > t_pru],
            ["Entrenamiento", "Validacion", "Prueba", "Excluido"],
            default="Sin fecha",
        )
        for p in ["Entrenamiento", "Validacion", "Prueba", "Excluido"]:
            sub_pd = df[(part == p) & cerrados_mask]
            sub_lgd = df[(part == p) & df["castigado"]]
            n_pd = len(sub_pd)
            n_malos = int((sub_pd["desenlace"] == "Malo (1)").sum())
            n_lgd = len(sub_lgd)
            n_rec = int((sub_lgd["recoveries"] > 0).sum())
            filas_est.append({
                "estrategia": nombre,
                "conjunto": p,
                "creditos_cerrados_pd": n_pd,
                "malos_pd": n_malos,
                "pct_malos_pd": round(n_malos / n_pd * 100, 2) if n_pd else 0.0,
                "castigados_lgd": n_lgd,
                "pct_castigados_con_rec": round(n_rec / n_lgd * 100, 2) if n_lgd else 0.0,
            })
    pd.DataFrame(filas_est).to_csv(out_res / "estrategias_particion.csv", index=False)
    return conteos_estado, mad


def generar_figura_censura(df: pd.DataFrame, mad: pd.DataFrame, out_fig: Path) -> None:
    resumen_anual = (
        df.groupby("anio")
        .agg(
            buenos=("desenlace", lambda s: (s == "Bueno (0)").sum()),
            malos=("desenlace", lambda s: (s == "Malo (1)").sum()),
            abiertos=("desenlace", lambda s: (s == "Abierto (NA)").sum()),
            total=("id", "size"),
        )
        .reset_index()
    )
    for c in ["buenos", "malos", "abiertos"]:
        resumen_anual[f"pct_{c}"] = resumen_anual[c] / resumen_anual["total"] * 100

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    fig.patch.set_facecolor(FONDO)

    # Panel A: Composición por año (censura)
    ax0 = axes[0]
    _estilo_ejes(ax0)
    anios = resumen_anual["anio"].astype(int).astype(str)
    b1 = resumen_anual["pct_buenos"]
    b2 = resumen_anual["pct_malos"]
    b3 = resumen_anual["pct_abiertos"]

    ax0.bar(anios, b1, color=AZUL, label="Pagados (Bueno = 0)")
    ax0.bar(anios, b2, bottom=b1, color=NARANJA, label="Incumplidos (Malo = 1)")
    ax0.bar(anios, b3, bottom=b1 + b2, color=GRIS_CLARO, alpha=0.75, label="En curso / Censurados (NA)")

    for i, val in enumerate(b3):
        if val >= 5:
            ax0.text(i, 100 - val / 2, f"{val:.0f}%\nNA", ha="center", va="center", fontsize=8, color=TEXTO, fontweight="bold")

    ax0.axvline(x=5.5, color=TEXTO, linestyle=":", linewidth=1.1)
    ax0.text(2.5, 103, "Entrenamiento (2007–2012)", ha="center", fontsize=8.5, color=TEXTO_2, fontweight="bold")
    ax0.text(6.0, 103, "Val", ha="center", fontsize=8.5, color=TEXTO_2, fontweight="bold")
    ax0.text(7.0, 103, "Prue", ha="center", fontsize=8.5, color=TEXTO_2, fontweight="bold")
    ax0.text(8.0, 103, "Excl.", ha="center", fontsize=8.5, color=NARANJA, fontweight="bold")

    ax0.set_ylim(0, 112)
    ax0.set_ylabel("Porcentaje de créditos originados (%)", color=TEXTO, fontsize=9.5)
    ax0.set_xlabel("Año de originación (issue_d)", color=TEXTO, fontsize=9.5)
    ax0.set_title("A. Maduración y censura por año de originación", color=TEXTO, fontsize=11, fontweight="bold", pad=10)
    ax0.legend(loc="lower left", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8)

    # Panel B: Tasa de malos sobre cerrados por plazo (36 vs 60 meses)
    ax1 = axes[1]
    _estilo_ejes(ax1)
    m36 = mad[mad["plazo_meses"] == 36].sort_values("anio")
    m60 = mad[mad["plazo_meses"] == 60].sort_values("anio")

    ax1.plot(m36["anio"].astype(int), m36["pct_malos_sobre_cerrados"], marker="o", linewidth=2.2, color=AZUL, label="Plazo 36 meses")
    ax1.plot(m60["anio"].astype(int), m60["pct_malos_sobre_cerrados"], marker="s", linewidth=2.2, color=NARANJA, label="Plazo 60 meses")

    # Anotar año 2010 (sin censura en ambos plazos)
    v36_2010 = float(m36.loc[m36["anio"] == 2010, "pct_malos_sobre_cerrados"].iloc[0])
    v60_2010 = float(m60.loc[m60["anio"] == 2010, "pct_malos_sobre_cerrados"].iloc[0])
    ax1.annotate(
        f"2010 sin censura:\n{v36_2010:.1f}% vs {v60_2010:.1f}%",
        xy=(2010, v60_2010),
        xytext=(2008.1, 31.5),
        arrowprops=dict(arrowstyle="->", color=TEXTO_2, lw=1.0),
        fontsize=8.5,
        color=TEXTO,
        bbox=dict(boxstyle="round,pad=0.3", fc=FONDO, ec=GUIA),
    )

    ax1.axvspan(2012.5, 2015.4, color=GRIS_CLARO, alpha=0.18, label="Zona con censura parcial/alta")
    ax1.set_ylim(5, 43)
    ax1.set_xticks(sorted(df["anio"].dropna().astype(int).unique()))
    ax1.set_ylabel("Tasa de incumplimiento sobre cerrados (%)", color=TEXTO, fontsize=9.5)
    ax1.set_xlabel("Año de originación (issue_d)", color=TEXTO, fontsize=9.5)
    ax1.set_title("B. Tasa de malos por plazo y efecto de la censura", color=TEXTO, fontsize=11, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8.5)

    fig.suptitle("Auditoría Temporal de Lending Club (887 379 créditos, 2007–2015)", color=TEXTO, fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_fig / "fig0_censura_y_tasa_malos_por_anio.png", dpi=180)
    plt.close(fig)


def prueba_sensibilidad_particion(out_res: Path) -> None:
    ruta_p1 = RAIZ_REPO / "parte_1" / "data" / "processed"
    if not (ruta_p1 / "pd_entrenamiento.parquet").exists():
        return
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
    from sklearn.neural_network import MLPClassifier

    tr = pd.read_parquet(ruta_p1 / "pd_entrenamiento.parquet")
    va = pd.read_parquet(ruta_p1 / "pd_validacion.parquet")
    te = pd.read_parquet(ruta_p1 / "pd_prueba.parquet")
    full = pd.concat([tr, va, te], ignore_index=True)
    cols = [c for c in tr.columns if c.startswith(("num__", "cat__"))]

    def eval_row(esquema: str, subconjunto: str, modelo_nombre: str, y, p) -> dict:
        auc = roc_auc_score(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        return {
            "esquema": esquema,
            "conjunto": subconjunto,
            "modelo": modelo_nombre,
            "n_creditos": int(len(y)),
            "tasa_malos_obs": round(float(np.mean(y)), 4),
            "pd_media_pred": round(float(np.mean(p)), 4),
            "auc": round(float(auc), 4),
            "gini": round(float(2 * auc - 1), 4),
            "ks": round(float(np.max(np.abs(tpr - fpr))), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
        }

    # Opción 1
    lr1 = LogisticRegression(max_iter=1000, random_state=2026, C=0.1).fit(tr[cols], tr["malo"].astype(int))
    mlp1 = MLPClassifier((64, 32), alpha=0.001, batch_size=256, early_stopping=True, random_state=2026, max_iter=150).fit(tr[cols], tr["malo"].astype(int))
    te_h1 = te[te["issue_d"] <= "2014-06-30"]

    # Opción 3
    tr3 = full[full["issue_d"] <= "2013-06-30"]
    va3 = full[(full["issue_d"] > "2013-06-30") & (full["issue_d"] <= "2013-12-31")]
    te3 = full[(full["issue_d"] > "2013-12-31") & (full["issue_d"] <= "2014-06-30")]
    lr3 = LogisticRegression(max_iter=1000, random_state=2026, C=0.1).fit(tr3[cols], tr3["malo"].astype(int))
    mlp3 = MLPClassifier((64, 32), alpha=0.001, batch_size=256, early_stopping=True, random_state=2026, max_iter=150).fit(tr3[cols], tr3["malo"].astype(int))

    filas = [
        eval_row("Opcion 1 (Train 2007-2012)", "Validacion 2013", "Regresion Logistica", va["malo"].astype(int), lr1.predict_proba(va[cols])[:, 1]),
        eval_row("Opcion 1 (Train 2007-2012)", "Validacion 2013", "MLP (64,32)", va["malo"].astype(int), mlp1.predict_proba(va[cols])[:, 1]),
        eval_row("Opcion 1 (Train 2007-2012)", "Prueba 2014", "Regresion Logistica", te["malo"].astype(int), lr1.predict_proba(te[cols])[:, 1]),
        eval_row("Opcion 1 (Train 2007-2012)", "Prueba 2014", "MLP (64,32)", te["malo"].astype(int), mlp1.predict_proba(te[cols])[:, 1]),
        eval_row("Opcion 1 (Train 2007-2012)", "Prueba 2014-H1", "Regresion Logistica", te_h1["malo"].astype(int), lr1.predict_proba(te_h1[cols])[:, 1]),
        eval_row("Opcion 1 (Train 2007-2012)", "Prueba 2014-H1", "MLP (64,32)", te_h1["malo"].astype(int), mlp1.predict_proba(te_h1[cols])[:, 1]),
        eval_row("Opcion 3 (Train 2007-2013H1)", "Validacion 2013-H2", "Regresion Logistica", va3["malo"].astype(int), lr3.predict_proba(va3[cols])[:, 1]),
        eval_row("Opcion 3 (Train 2007-2013H1)", "Validacion 2013-H2", "MLP (64,32)", va3["malo"].astype(int), mlp3.predict_proba(va3[cols])[:, 1]),
        eval_row("Opcion 3 (Train 2007-2013H1)", "Prueba 2014-H1", "Regresion Logistica", te3["malo"].astype(int), lr3.predict_proba(te3[cols])[:, 1]),
        eval_row("Opcion 3 (Train 2007-2013H1)", "Prueba 2014-H1", "MLP (64,32)", te3["malo"].astype(int), mlp3.predict_proba(te3[cols])[:, 1]),
    ]
    pd.DataFrame(filas).to_csv(out_res / "comparacion_opcion1_vs_opcion3.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoría temporal del dataset Lending Club")
    parser.add_argument("--datos", default=str(RAIZ_REPO / "data" / "raw" / "loan.csv"))
    parser.add_argument("--muestra", type=float, default=None)
    parser.add_argument("--omitir_sensibilidad", action="store_true")
    args = parser.parse_args()

    out_res = DIR_ACTUAL / "resultados"
    out_fig = DIR_ACTUAL / "figuras"
    out_res.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo {args.datos} ...")
    df = cargar_crudo_particion(Path(args.datos), args.muestra)
    _, mad = generar_tablas_auditoria(df, out_res)
    generar_figura_censura(df, mad, out_fig)

    if not args.omitir_sensibilidad and args.muestra is None:
        print("Ejecutando comparación de sensibilidad Opción 1 vs Opción 3...")
        prueba_sensibilidad_particion(out_res)

    print(f"Auditoría completada. Resultados en {out_res} y figuras en {out_fig}")


if __name__ == "__main__":
    main()
