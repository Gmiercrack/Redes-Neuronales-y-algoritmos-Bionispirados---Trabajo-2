"""
Análisis Descriptivo y Contraste de Hipótesis sobre Datos Crudos (Parte 1 - PD).

Lee directamente data/raw/loan.csv (ANTES del preprocesamiento y de los modelos),
filtra exclusivamente el periodo de entrenamiento con desenlace conocido (2007–2012,
90 347 créditos) para no contaminar la exploración con validación ni prueba, y:
  0. Diagnostica faltantes, valores extremos y categorías raras (justifica pd.yaml).
  1. Evalúa H1: Grado asignado (grade) y tasa de interés (int_rate).
  2. Evalúa H2: Capacidad de pago (dti e ingreso anual annual_inc).
  3. Evalúa H3: Estrés de liquidez (revol_util) y señales de buró (inq, delinq, pub_rec).
  4. Evalúa H4: Plazo contractual (36 vs 60 meses, global y en cohorte madura 2010) y purpose.
  5. Evalúa H5 (Contraste): Verificación de ingresos, antigüedad laboral y vivienda.

Uso (desde la raíz del repositorio):
    .\\.venv\\Scripts\\python.exe Analisis_del_problema/analisis_descriptivo/descriptivo.py
Modo demostración (< 2 minutos):
    .\\.venv\\Scripts\\python.exe Analisis_del_problema/analisis_descriptivo/descriptivo.py --muestra 0.1
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
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr

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

COLUMNAS_CRUDAS = [
    "id", "issue_d", "loan_status",
    "loan_amnt", "funded_amnt", "int_rate", "installment",
    "annual_inc", "dti", "delinq_2yrs", "inq_last_6mths",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "term", "emp_length", "earliest_cr_line",
    "grade", "home_ownership", "verification_status", "purpose", "initial_list_status",
]

VARS_NUMERICAS = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "revol_bal",
    "revol_util", "total_acc", "plazo_meses", "antiguedad_empleo",
    "meses_historial_credito",
]
VARS_CATEGORICAS = [
    "grade", "home_ownership", "verification_status", "purpose", "initial_list_status",
]

# Paleta editorial compartida con Parte 2 (referencia_lgd.py)
AZUL, NARANJA = "#2a78d6", "#eb6834"
GRIS_CLARO = "#b8b6b0"
TEXTO, TEXTO_2, FONDO, GUIA = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d4"


def _estilo_ejes(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(FONDO)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GUIA)
    ax.spines["bottom"].set_color(GUIA)
    ax.tick_params(colors=TEXTO_2, labelsize=9)
    ax.grid(axis=grid_axis, color=GUIA, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def a_fecha_mes(serie: pd.Series) -> pd.Series:
    fechas = pd.to_datetime(serie, format="%b-%Y", errors="coerce")
    if fechas.isna().mean() > 0.5:
        fechas = pd.to_datetime(serie, errors="coerce", format="mixed")
    return fechas


def a_numero(serie: pd.Series) -> pd.Series:
    if serie.dtype == object:
        serie = serie.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(serie, errors="coerce")


def cargar_entrenamiento_crudo(
    ruta_csv: Path,
    fin_entrenamiento: str = "2012-12-31",
    muestra: float | None = None,
    semilla: int = 2026,
) -> pd.DataFrame:
    """Lee loan.csv sin imputar ni escalar y filtra la ventana de entrenamiento."""
    df = pd.read_csv(ruta_csv, usecols=COLUMNAS_CRUDAS, low_memory=False)
    if muestra:
        df = df.sample(frac=muestra, random_state=semilla).reset_index(drop=True)

    df["issue_d"] = a_fecha_mes(df["issue_d"])
    df["loan_status"] = df["loan_status"].astype(str).str.strip()

    df["malo"] = pd.Series(pd.NA, index=df.index, dtype="Int8")
    df.loc[df["loan_status"].isin(ESTADOS_BUENOS), "malo"] = 0
    df.loc[df["loan_status"].isin(ESTADOS_MALOS), "malo"] = 1

    corte = pd.Timestamp(fin_entrenamiento)
    df = df[(df["issue_d"] <= corte) & df["malo"].notna()].copy()
    df["malo"] = df["malo"].astype(int)
    df["anio"] = df["issue_d"].dt.year

    # Solo limpieza de formato (unidades originales intactas, sin imputación)
    df["plazo_meses"] = a_numero(df["term"].astype(str).str.extract(r"(\d+)")[0])
    texto_emp = df["emp_length"].astype(str)
    df["antiguedad_empleo"] = np.where(
        texto_emp.str.contains("<"), 0, a_numero(texto_emp.str.extract(r"(\d+)")[0])
    )
    inicio_cr = a_fecha_mes(df["earliest_cr_line"])
    df["meses_historial_credito"] = (
        (df["issue_d"].dt.year - inicio_cr.dt.year) * 12
        + (df["issue_d"].dt.month - inicio_cr.dt.month)
    )

    for col in VARS_NUMERICAS:
        df[col] = a_numero(df[col])
    for col in VARS_CATEGORICAS:
        df[col] = df[col].astype("string").str.strip()

    return df.reset_index(drop=True)


def _tabla_tasa(df: pd.DataFrame, col: str, orden: list | None = None) -> pd.DataFrame:
    t = (
        df.groupby(col, observed=False)
        .agg(creditos=("malo", "size"), malos=("malo", "sum"))
        .reset_index()
    )
    t["tasa_malos"] = t["malos"] / t["creditos"]
    t["pct_tasa"] = t["tasa_malos"] * 100
    # Error estándar binomial para intervalos de confianza 95 %
    t["ic95_pp"] = 1.96 * np.sqrt(t["tasa_malos"] * (1 - t["tasa_malos"]) / t["creditos"].clip(lower=1)) * 100
    if orden is not None:
        t[col] = pd.Categorical(t[col], categories=orden, ordered=True)
        t = t.sort_values(col).reset_index(drop=True)
    return t


def _chi2_p(df: pd.DataFrame, col: str) -> float:
    ct = pd.crosstab(df[col], df["malo"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 1.0
    _, p, _, _ = chi2_contingency(ct)
    return float(p)


# =============================================================================
# 0. Diagnóstico crudo (justificación directa de las reglas de pd.yaml)
# =============================================================================
def diagnosticar_datos_para_preprocesamiento(df: pd.DataFrame, out_res: Path) -> pd.DataFrame:
    filas = []
    n = len(df)
    for col in VARS_NUMERICAS:
        s = df[col]
        n_na = int(s.isna().sum())
        filas.append({
            "variable": col,
            "tipo": "numerica",
            "n_validos": n - n_na,
            "n_faltantes": n_na,
            "pct_faltantes": round(n_na / n * 100, 3),
            "media": round(float(s.mean()), 3) if n - n_na else np.nan,
            "std": round(float(s.std()), 3) if n - n_na else np.nan,
            "mediana": round(float(s.median()), 3) if n - n_na else np.nan,
            "p95": round(float(s.quantile(0.95)), 3) if n - n_na else np.nan,
            "p99": round(float(s.quantile(0.99)), 3) if n - n_na else np.nan,
            "max": round(float(s.max()), 3) if n - n_na else np.nan,
            "nota_preprocesamiento": "Indicador faltante + mediana" if n_na > 0 else "Sin faltantes",
        })
    for col in VARS_CATEGORICAS:
        s = df[col].dropna()
        freqs = s.value_counts(normalize=True)
        infreq = freqs[freqs < 0.005].index.tolist()
        n_na = int(df[col].isna().sum())
        filas.append({
            "variable": col,
            "tipo": "categorica",
            "n_validos": int(len(s)),
            "n_faltantes": n_na,
            "pct_faltantes": round(n_na / n * 100, 3),
            "media": np.nan,
            "std": np.nan,
            "mediana": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": float(len(freqs)),
            "nota_preprocesamiento": f"{len(freqs)} categorias; {len(infreq)} infrecuentes (<0.5%): {infreq}",
        })
    diag = pd.DataFrame(filas)
    diag.to_csv(out_res / "diagnostico_variables_crudas.csv", index=False)
    return diag


# =============================================================================
# H1: Calificación (grade) y tasa de interés (int_rate)
# =============================================================================
def evaluar_h1_grado_y_tasa(df: pd.DataFrame, out_res: Path, out_fig: Path) -> dict:
    tasa_base = df["malo"].mean() * 100
    grados = ["A", "B", "C", "D", "E", "F", "G"]
    t_grade = _tabla_tasa(df, "grade", orden=grados)
    t_grade["variable"] = "grade"
    t_grade["int_rate_medio"] = df.groupby("grade", observed=False)["int_rate"].mean().reindex(grados).values

    # Tramos de tasa de interés
    bins_ir = [0, 8, 11, 14, 17, 20, 100]
    labels_ir = ["< 8%", "8–11%", "11–14%", "14–17%", "17–20%", ">= 20%"]
    df_tmp = df.copy()
    df_tmp["tramo_int_rate"] = pd.cut(df_tmp["int_rate"], bins=bins_ir, labels=labels_ir, right=False)
    t_ir = _tabla_tasa(df_tmp, "tramo_int_rate", orden=labels_ir)
    t_ir["variable"] = "int_rate"

    tabla_out = pd.concat([
        t_grade.rename(columns={"grade": "categoria"}),
        t_ir.rename(columns={"tramo_int_rate": "categoria"}),
    ], ignore_index=True)
    tabla_out.to_csv(out_res / "h1_grado_y_tasa.csv", index=False)

    # Pruebas estadísticas
    p_chi_grade = _chi2_p(df, "grade")
    rho_ir, p_spear_ir = spearmanr(df["int_rate"], df["malo"], nan_policy="omit")
    monotona_grade = bool(t_grade["pct_tasa"].is_monotonic_increasing)

    # Figura 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    fig.patch.set_facecolor(FONDO)

    ax0 = axes[0]
    _estilo_ejes(ax0)
    colores_g = [AZUL if v <= tasa_base else NARANJA for v in t_grade["pct_tasa"]]
    bars = ax0.bar(t_grade["grade"].astype(str), t_grade["pct_tasa"], yerr=t_grade["ic95_pp"],
                   color=colores_g, capsize=3, error_kw=dict(ecolor=TEXTO_2, lw=1))
    ax0.axhline(tasa_base, color=TEXTO, linestyle="--", linewidth=1, label=f"Promedio ({tasa_base:.1f}%)")
    for b, n_c in zip(bars, t_grade["creditos"]):
        ax0.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{b.get_height():.1f}%\n(n={n_c:,})",
                 ha="center", va="bottom", fontsize=7.5, color=TEXTO)
    ax0.set_ylim(0, t_grade["pct_tasa"].max() * 1.24)
    ax0.set_xlabel("Grado asignado por Lending Club (grade)", color=TEXTO)
    ax0.set_ylabel("Tasa de incumplimiento (%)", color=TEXTO)
    ax0.set_title("A. Tasa de incumplimiento por grado de riesgo", color=TEXTO, fontweight="bold")
    ax0.legend(loc="upper left", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8.5)

    ax1 = axes[1]
    _estilo_ejes(ax1)
    ax1.errorbar(range(len(t_ir)), t_ir["pct_tasa"], yerr=t_ir["ic95_pp"], marker="o",
                 color=NARANJA, linewidth=2.2, capsize=4, markerfacecolor=FONDO, markeredgewidth=2)
    ax1.axhline(tasa_base, color=TEXTO, linestyle="--", linewidth=1, label=f"Promedio ({tasa_base:.1f}%)")
    ax1.set_xticks(range(len(t_ir)))
    ax1.set_xticklabels(t_ir["tramo_int_rate"].astype(str))
    for i, (pct, n_c) in enumerate(zip(t_ir["pct_tasa"], t_ir["creditos"])):
        ax1.text(i, pct + 1.3, f"{pct:.1f}%", ha="center", fontsize=8, color=TEXTO, fontweight="bold")
    ax1.set_ylim(0, t_ir["pct_tasa"].max() * 1.22)
    ax1.set_xlabel("Rango de tasa de interés (int_rate)", color=TEXTO)
    ax1.set_ylabel("Tasa de incumplimiento (%)", color=TEXTO)
    ax1.set_title("B. Tasa de incumplimiento por rango de tasa de interés", color=TEXTO, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8.5)

    fig.suptitle("H1: Tarificación y Grado vs. Probabilidad de Incumplimiento (Entrenamiento 2007–2012)",
                 color=TEXTO, fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_fig / "fig1_h1_grado_y_tasa.png", dpi=180)
    plt.close(fig)

    return {
        "hipotesis": "H1: Grado y tasa de interes",
        "veredicto": "CONFIRMADA" if monotona_grade and p_chi_grade < 1e-6 else "PARCIAL",
        "monotona_grado_A_G": monotona_grade,
        "tasa_grado_A_pct": round(float(t_grade.loc[t_grade["grade"] == "A", "pct_tasa"].iloc[0]), 2),
        "tasa_grado_G_pct": round(float(t_grade.loc[t_grade["grade"] == "G", "pct_tasa"].iloc[0]), 2),
        "spearman_int_rate": round(float(rho_ir), 4),
        "p_value_chi2_grade": p_chi_grade,
        "p_value_spearman_int_rate": float(p_spear_ir),
    }


# =============================================================================
# H2: Capacidad de pago (dti e ingreso anual)
# =============================================================================
def evaluar_h2_capacidad_pago(df: pd.DataFrame, out_res: Path, out_fig: Path) -> dict:
    tasa_base = df["malo"].mean() * 100
    df_tmp = df.copy()

    bins_dti = [-0.1, 10, 15, 20, 25, 30, 100]
    labels_dti = ["< 10%", "10–15%", "15–20%", "20–25%", "25–30%", ">= 30%"]
    df_tmp["tramo_dti"] = pd.cut(df_tmp["dti"], bins=bins_dti, labels=labels_dti)
    t_dti = _tabla_tasa(df_tmp.dropna(subset=["tramo_dti"]), "tramo_dti", orden=labels_dti)
    t_dti["variable"] = "dti"

    # Quintiles de annual_inc en USD reales
    q_inc, edges = pd.qcut(df_tmp["annual_inc"], q=5, retbins=True, duplicates="drop")
    labels_inc = [f"Q{i+1}\n(${edges[i]/1000:.0f}k–${edges[i+1]/1000:.0f}k)" for i in range(len(edges) - 1)]
    df_tmp["quintil_ingreso"] = pd.qcut(df_tmp["annual_inc"], q=5, labels=labels_inc, duplicates="drop")
    t_inc = _tabla_tasa(df_tmp.dropna(subset=["quintil_ingreso"]), "quintil_ingreso", orden=labels_inc)
    t_inc["variable"] = "annual_inc"

    tabla_out = pd.concat([
        t_dti.rename(columns={"tramo_dti": "categoria"}),
        t_inc.rename(columns={"quintil_ingreso": "categoria"}),
    ], ignore_index=True)
    tabla_out["categoria"] = tabla_out["categoria"].astype(str).str.replace("\n", " ")
    tabla_out.to_csv(out_res / "h2_dti_e_ingreso.csv", index=False)

    # Mann-Whitney U entre buenos y malos
    _, p_mw_dti = mannwhitneyu(df.loc[df["malo"] == 1, "dti"].dropna(), df.loc[df["malo"] == 0, "dti"].dropna())
    _, p_mw_inc = mannwhitneyu(df.loc[df["malo"] == 1, "annual_inc"].dropna(), df.loc[df["malo"] == 0, "annual_inc"].dropna())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    fig.patch.set_facecolor(FONDO)

    ax0 = axes[0]
    _estilo_ejes(ax0)
    colores_dti = [AZUL if v <= tasa_base else NARANJA for v in t_dti["pct_tasa"]]
    bars0 = ax0.bar(t_dti["tramo_dti"].astype(str), t_dti["pct_tasa"], yerr=t_dti["ic95_pp"], color=colores_dti, capsize=3)
    ax0.axhline(tasa_base, color=TEXTO, linestyle="--", linewidth=1, label=f"Promedio ({tasa_base:.1f}%)")
    for b, pct in zip(bars0, t_dti["pct_tasa"]):
        ax0.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{pct:.1f}%", ha="center", fontsize=8, color=TEXTO)
    ax0.set_ylim(0, t_dti["pct_tasa"].max() * 1.22)
    ax0.set_xlabel("Relación deuda mensual / ingreso (dti)", color=TEXTO)
    ax0.set_ylabel("Tasa de incumplimiento (%)", color=TEXTO)
    ax0.set_title("A. Tasa de incumplimiento según nivel de DTI", color=TEXTO, fontweight="bold")
    ax0.legend(loc="upper left", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8.5)

    ax1 = axes[1]
    _estilo_ejes(ax1)
    colores_inc = [NARANJA if v > tasa_base else AZUL for v in t_inc["pct_tasa"]]
    bars1 = ax1.bar(range(len(t_inc)), t_inc["pct_tasa"], yerr=t_inc["ic95_pp"], color=colores_inc, capsize=3)
    ax1.axhline(tasa_base, color=TEXTO, linestyle="--", linewidth=1, label=f"Promedio ({tasa_base:.1f}%)")
    ax1.set_xticks(range(len(t_inc)))
    ax1.set_xticklabels(t_inc["quintil_ingreso"].astype(str), fontsize=8)
    for b, pct in zip(bars1, t_inc["pct_tasa"]):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{pct:.1f}%", ha="center", fontsize=8, color=TEXTO)
    ax1.set_ylim(0, t_inc["pct_tasa"].max() * 1.25)
    ax1.set_xlabel("Quintiles de ingreso anual declarado (annual_inc)", color=TEXTO)
    ax1.set_ylabel("Tasa de incumplimiento (%)", color=TEXTO)
    ax1.set_title("B. Tasa de incumplimiento por quintil de ingreso", color=TEXTO, fontweight="bold")
    ax1.legend(loc="upper right", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8.5)

    fig.suptitle("H2: Capacidad de Pago (DTI e Ingreso Anual) vs. Incumplimiento (2007–2012)",
                 color=TEXTO, fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_fig / "fig2_h2_dti_e_ingreso.png", dpi=180)
    plt.close(fig)

    return {
        "hipotesis": "H2: Capacidad de pago (DTI e ingreso anual)",
        "veredicto": "CONFIRMADA",
        "dti_bajo_pct": round(float(t_dti["pct_tasa"].iloc[0]), 2),
        "dti_alto_pct": round(float(t_dti["pct_tasa"].iloc[-1]), 2),
        "ingreso_q1_pct": round(float(t_inc["pct_tasa"].iloc[0]), 2),
        "ingreso_q5_pct": round(float(t_inc["pct_tasa"].iloc[-1]), 2),
        "p_value_mannwhitney_dti": float(p_mw_dti),
        "p_value_mannwhitney_annual_inc": float(p_mw_inc),
    }


# =============================================================================
# H3: Liquidez en tarjetas y señales del buró
# =============================================================================
def evaluar_h3_liquidez_y_buro(df: pd.DataFrame, out_res: Path, out_fig: Path) -> dict:
    tasa_base = df["malo"].mean() * 100
    df_tmp = df.copy()

    bins_ru = [-0.1, 25, 50, 75, 90, 2000]
    labels_ru = ["< 25%", "25–50%", "50–75%", "75–90%", ">= 90%"]
    df_tmp["tramo_revol_util"] = pd.cut(df_tmp["revol_util"], bins=bins_ru, labels=labels_ru)
    t_ru = _tabla_tasa(df_tmp.dropna(subset=["tramo_revol_util"]), "tramo_revol_util", orden=labels_ru)
    t_ru["variable"] = "revol_util"

    df_tmp["grupo_inq"] = np.where(df_tmp["inq_last_6mths"] >= 3, "3+", df_tmp["inq_last_6mths"].fillna(-1).astype(int).astype(str))
    df_tmp.loc[df_tmp["inq_last_6mths"].isna(), "grupo_inq"] = np.nan
    t_inq = _tabla_tasa(df_tmp.dropna(subset=["grupo_inq"]), "grupo_inq", orden=["0", "1", "2", "3+"])
    t_inq["variable"] = "inq_last_6mths"

    df_tmp["grupo_delinq"] = np.where(df_tmp["delinq_2yrs"] >= 2, "2+", df_tmp["delinq_2yrs"].fillna(-1).astype(int).astype(str))
    df_tmp.loc[df_tmp["delinq_2yrs"].isna(), "grupo_delinq"] = np.nan
    t_del = _tabla_tasa(df_tmp.dropna(subset=["grupo_delinq"]), "grupo_delinq", orden=["0", "1", "2+"])
    t_del["variable"] = "delinq_2yrs"

    df_tmp["grupo_pub"] = np.where(df_tmp["pub_rec"] >= 1, "1+", "0")
    df_tmp.loc[df_tmp["pub_rec"].isna(), "grupo_pub"] = np.nan
    t_pub = _tabla_tasa(df_tmp.dropna(subset=["grupo_pub"]), "grupo_pub", orden=["0", "1+"])
    t_pub["variable"] = "pub_rec"

    tabla_out = pd.concat([
        t_ru.rename(columns={"tramo_revol_util": "categoria"}),
        t_inq.rename(columns={"grupo_inq": "categoria"}),
        t_del.rename(columns={"grupo_delinq": "categoria"}),
        t_pub.rename(columns={"grupo_pub": "categoria"}),
    ], ignore_index=True)
    tabla_out.to_csv(out_res / "h3_liquidez_y_buro.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=180)
    fig.patch.set_facecolor(FONDO)

    ax0 = axes[0]
    _estilo_ejes(ax0)
    colores_ru = [AZUL if v <= tasa_base else NARANJA for v in t_ru["pct_tasa"]]
    bars0 = ax0.bar(t_ru["tramo_revol_util"].astype(str), t_ru["pct_tasa"], yerr=t_ru["ic95_pp"], color=colores_ru, capsize=3)
    ax0.axhline(tasa_base, color=TEXTO, linestyle="--", linewidth=1, label=f"Promedio ({tasa_base:.1f}%)")
    for b, pct in zip(bars0, t_ru["pct_tasa"]):
        ax0.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{pct:.1f}%", ha="center", fontsize=8, color=TEXTO)
    ax0.set_ylim(0, t_ru["pct_tasa"].max() * 1.24)
    ax0.set_xlabel("Utilización de cupo rotativo / tarjetas (revol_util)", color=TEXTO)
    ax0.set_ylabel("Tasa de incumplimiento (%)", color=TEXTO)
    ax0.set_title("A. Estrés de liquidez por uso de tarjetas (revol_util)", color=TEXTO, fontweight="bold")
    ax0.legend(loc="upper left", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8.5)

    ax1 = axes[1]
    _estilo_ejes(ax1)
    etiquetas = [f"Consultas\n{c}" for c in t_inq["grupo_inq"]] + [f"Moras 2a\n{c}" for c in t_del["grupo_delinq"]] + [f"Reg. púb.\n{c}" for c in t_pub["grupo_pub"]]
    valores = list(t_inq["pct_tasa"]) + list(t_del["pct_tasa"]) + list(t_pub["pct_tasa"])
    errores = list(t_inq["ic95_pp"]) + list(t_del["ic95_pp"]) + list(t_pub["ic95_pp"])
    colores_b = [AZUL if v <= tasa_base else NARANJA for v in valores]
    bars1 = ax1.bar(range(len(etiquetas)), valores, yerr=errores, color=colores_b, capsize=3)
    ax1.axhline(tasa_base, color=TEXTO, linestyle="--", linewidth=1, label=f"Promedio ({tasa_base:.1f}%)")
    ax1.axvline(3.5, color=GUIA, linestyle="-", linewidth=1)
    ax1.axvline(6.5, color=GUIA, linestyle="-", linewidth=1)
    ax1.set_xticks(range(len(etiquetas)))
    ax1.set_xticklabels(etiquetas, fontsize=7.5)
    for b, pct in zip(bars1, valores):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7, f"{pct:.1f}%", ha="center", fontsize=7.5, color=TEXTO)
    ax1.set_ylim(0, max(valores) * 1.25)
    ax1.set_ylabel("Tasa de incumplimiento (%)", color=TEXTO)
    ax1.set_title("B. Alertas recientes en buró de crédito", color=TEXTO, fontweight="bold")

    fig.suptitle("H3: Uso de Tarjetas Rotativas y Señales de Buró vs. Incumplimiento (2007–2012)",
                 color=TEXTO, fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_fig / "fig3_h3_liquidez_y_buro.png", dpi=180)
    plt.close(fig)

    return {
        "hipotesis": "H3: Liquidez (revol_util) y alertas en buro",
        "veredicto": "CONFIRMADA",
        "revol_util_menor_25_pct": round(float(t_ru["pct_tasa"].iloc[0]), 2),
        "revol_util_mayor_90_pct": round(float(t_ru["pct_tasa"].iloc[-1]), 2),
        "inq_0_pct": round(float(t_inq["pct_tasa"].iloc[0]), 2),
        "inq_3mas_pct": round(float(t_inq["pct_tasa"].iloc[-1]), 2),
        "pub_rec_0_pct": round(float(t_pub["pct_tasa"].iloc[0]), 2),
        "pub_rec_1mas_pct": round(float(t_pub["pct_tasa"].iloc[-1]), 2),
        "p_value_chi2_inq": _chi2_p(df_tmp.dropna(subset=["grupo_inq"]), "grupo_inq"),
        "p_value_chi2_pub_rec": _chi2_p(df_tmp.dropna(subset=["grupo_pub"]), "grupo_pub"),
    }


# =============================================================================
# H4: Plazo del préstamo (36 vs 60 meses) y propósito
# =============================================================================
def evaluar_h4_plazo_y_proposito(df: pd.DataFrame, out_res: Path, out_fig: Path) -> dict:
    tasa_base = df["malo"].mean() * 100
    t_plazo_all = _tabla_tasa(df, "plazo_meses", orden=[36, 60])
    t_plazo_all["cohorte"] = "2007-2012 (todo entrenamiento)"

    df_2010 = df[df["anio"] == 2010]
    t_plazo_2010 = _tabla_tasa(df_2010, "plazo_meses", orden=[36, 60])
    t_plazo_2010["cohorte"] = "2010 (sin censura en 36m ni 60m)"

    t_purp = _tabla_tasa(df, "purpose").sort_values("pct_tasa", ascending=True).reset_index(drop=True)
    t_purp["cohorte"] = "2007-2012 (por proposito)"

    tabla_out = pd.concat([
        t_plazo_all.rename(columns={"plazo_meses": "categoria"}),
        t_plazo_2010.rename(columns={"plazo_meses": "categoria"}),
        t_purp.rename(columns={"purpose": "categoria"}),
    ], ignore_index=True)
    tabla_out.to_csv(out_res / "h4_plazo_y_proposito.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180, gridspec_kw={"width_ratios": [1, 1.4]})
    fig.patch.set_facecolor(FONDO)

    # Panel A: Plazo 36 vs 60 (Entrenamiento total vs 2010 maduro)
    ax0 = axes[0]
    _estilo_ejes(ax0)
    x = np.arange(2)
    w = 0.34
    v_all = t_plazo_all["pct_tasa"].values
    v_2010 = t_plazo_2010["pct_tasa"].values
    b1 = ax0.bar(x - w / 2, v_all, width=w, color=AZUL, label="Todo entrenamiento (2007–2012)")
    b2 = ax0.bar(x + w / 2, v_2010, width=w, color=NARANJA, label="Cohorte 2010 (ciclo 100% cerrado)")
    for b in list(b1) + list(b2):
        ax0.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}%", ha="center", fontsize=8.5, color=TEXTO, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(["36 meses", "60 meses"], fontsize=10)
    ax0.set_ylim(0, max(max(v_all), max(v_2010)) * 1.25)
    ax0.set_ylabel("Tasa de incumplimiento (%)", color=TEXTO)
    ax0.set_title("A. Impacto del plazo (controlando por censura)", color=TEXTO, fontweight="bold")
    ax0.legend(loc="upper left", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8)

    # Panel B: Ranking de propósitos
    ax1 = axes[1]
    _estilo_ejes(ax1, grid_axis="x")
    colores_p = [NARANJA if v > tasa_base else AZUL for v in t_purp["pct_tasa"]]
    bars1 = ax1.barh(t_purp["purpose"].astype(str), t_purp["pct_tasa"], color=colores_p)
    ax1.axvline(tasa_base, color=TEXTO, linestyle="--", linewidth=1, label=f"Promedio ({tasa_base:.1f}%)")
    for b, pct, n_c in zip(bars1, t_purp["pct_tasa"], t_purp["creditos"]):
        ax1.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, f"{pct:.1f}% (n={n_c:,})",
                 va="center", fontsize=7.5, color=TEXTO)
    ax1.set_xlim(0, t_purp["pct_tasa"].max() * 1.28)
    ax1.set_xlabel("Tasa de incumplimiento (%)", color=TEXTO)
    ax1.set_title("B. Riesgo según propósito del préstamo (purpose)", color=TEXTO, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True, facecolor=FONDO, edgecolor=GUIA, fontsize=8)

    fig.suptitle("H4: Plazo Contractual y Propósito del Crédito vs. Incumplimiento",
                 color=TEXTO, fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_fig / "fig4_h4_plazo_y_proposito.png", dpi=180)
    plt.close(fig)

    return {
        "hipotesis": "H4: Plazo contractual (36 vs 60m) y proposito",
        "veredicto": "CONFIRMADA",
        "tasa_36m_total_pct": round(float(v_all[0]), 2),
        "tasa_60m_total_pct": round(float(v_all[1]), 2),
        "tasa_36m_2010_pct": round(float(v_2010[0]), 2),
        "tasa_60m_2010_pct": round(float(v_2010[1]), 2),
        "proposito_mas_riesgoso": str(t_purp.iloc[-1]["purpose"]),
        "proposito_mas_riesgoso_pct": round(float(t_purp.iloc[-1]["pct_tasa"]), 2),
        "proposito_menos_riesgoso": str(t_purp.iloc[0]["purpose"]),
        "proposito_menos_riesgoso_pct": round(float(t_purp.iloc[0]["pct_tasa"]), 2),
        "p_value_chi2_plazo_2010": _chi2_p(df_2010, "plazo_meses"),
        "p_value_chi2_purpose": _chi2_p(df, "purpose"),
    }


# =============================================================================
# H5 (Contraste): Verificación de ingresos, antigüedad laboral y vivienda
# =============================================================================
def evaluar_h5_estabilidad_y_verificacion(df: pd.DataFrame, out_res: Path, out_fig: Path) -> dict:
    tasa_base = df["malo"].mean() * 100
    df_tmp = df.copy()

    t_ver = _tabla_tasa(df_tmp, "verification_status", orden=["Not Verified", "Source Verified", "Verified"])
    t_ver["variable"] = "verification_status"

    # Antigüedad laboral en tramos claros + Faltante
    df_tmp["tramo_emp"] = np.where(
        df_tmp["antiguedad_empleo"].isna(), "Sin dato",
        np.where(df_tmp["antiguedad_empleo"] == 0, "< 1 año",
        np.where(df_tmp["antiguedad_empleo"] <= 3, "1–3 años",
        np.where(df_tmp["antiguedad_empleo"] <= 6, "4–6 años",
        np.where(df_tmp["antiguedad_empleo"] <= 9, "7–9 años", "10+ años"))))
    )
    orden_emp = ["< 1 año", "1–3 años", "4–6 años", "7–9 años", "10+ años", "Sin dato"]
    t_emp = _tabla_tasa(df_tmp, "tramo_emp", orden=orden_emp)
    t_emp["variable"] = "antiguedad_empleo"

    # Vivienda (principales categorías)
    principales_viv = ["MORTGAGE", "OWN", "RENT"]
    df_tmp["viv_grupo"] = np.where(df_tmp["home_ownership"].isin(principales_viv), df_tmp["home_ownership"], "OTHER/RARE")
    t_viv = _tabla_tasa(df_tmp, "viv_grupo", orden=["MORTGAGE", "OWN", "RENT", "OTHER/RARE"])
    t_viv["variable"] = "home_ownership"

    tabla_out = pd.concat([
        t_ver.rename(columns={"verification_status": "categoria"}),
        t_emp.rename(columns={"tramo_emp": "categoria"}),
        t_viv.rename(columns={"viv_grupo": "categoria"}),
    ], ignore_index=True)
    tabla_out.to_csv(out_res / "h5_verificacion_empleo_vivienda.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), dpi=180)
    fig.patch.set_facecolor(FONDO)

    panels = [
        (axes[0], t_ver, "verification_status", "A. Verificación de ingresos\n(Paradoja de selección)", "Estado de verificación"),
        (axes[1], t_emp, "tramo_emp", "B. Antigüedad laboral\n(Efecto muy débil + faltantes)", "Años en el empleo"),
        (axes[2], t_viv, "viv_grupo", "C. Tenencia de vivienda\n(Arriendo vs. Hipoteca/Propia)", "Tipo de vivienda"),
    ]
    for ax, tab, col, tit, xlab in panels:
        _estilo_ejes(ax)
        cols_bar = [NARANJA if v > tasa_base else AZUL for v in tab["pct_tasa"]]
        bars = ax.bar(tab[col].astype(str), tab["pct_tasa"], yerr=tab["ic95_pp"], color=cols_bar, capsize=3)
        ax.axhline(tasa_base, color=TEXTO, linestyle="--", linewidth=1)
        for b, pct in zip(bars, tab["pct_tasa"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{pct:.1f}%", ha="center", fontsize=7.8, color=TEXTO)
        ax.set_ylim(0, max(tab["pct_tasa"].max() * 1.25, tasa_base * 1.35))
        ax.set_title(tit, color=TEXTO, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlab, color=TEXTO, fontsize=9)
        ax.tick_params(axis="x", rotation=22)
    axes[0].set_ylabel("Tasa de incumplimiento (%)", color=TEXTO)

    fig.suptitle("H5 (Contraste): Verificación de Ingresos, Antigüedad Laboral y Vivienda (2007–2012)",
                 color=TEXTO, fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_fig / "fig5_h5_verificacion_empleo_vivienda.png", dpi=180)
    plt.close(fig)

    tasa_not_ver = float(t_ver.loc[t_ver["verification_status"] == "Not Verified", "pct_tasa"].iloc[0])
    tasa_ver = float(t_ver.loc[t_ver["verification_status"] == "Verified", "pct_tasa"].iloc[0])

    return {
        "hipotesis": "H5 (Contraste): Verificacion de ingresos, empleo y vivienda",
        "veredicto": "REFUTADA PARCIALMENTE (paradoja en verificacion y efecto debil en empleo)",
        "not_verified_pct": round(tasa_not_ver, 2),
        "verified_pct": round(tasa_ver, 2),
        "empleo_menor_1a_pct": round(float(t_emp.loc[t_emp["tramo_emp"] == "< 1 año", "pct_tasa"].iloc[0]), 2),
        "empleo_10mas_pct": round(float(t_emp.loc[t_emp["tramo_emp"] == "10+ años", "pct_tasa"].iloc[0]), 2),
        "empleo_sin_dato_pct": round(float(t_emp.loc[t_emp["tramo_emp"] == "Sin dato", "pct_tasa"].iloc[0]), 2),
        "mortgage_pct": round(float(t_viv.loc[t_viv["viv_grupo"] == "MORTGAGE", "pct_tasa"].iloc[0]), 2),
        "rent_pct": round(float(t_viv.loc[t_viv["viv_grupo"] == "RENT", "pct_tasa"].iloc[0]), 2),
        "p_value_chi2_verification": _chi2_p(df_tmp, "verification_status"),
        "p_value_chi2_home": _chi2_p(df_tmp, "viv_grupo"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis descriptivo y contraste de hipótesis (PD)")
    parser.add_argument("--datos", default=str(RAIZ_REPO / "data" / "raw" / "loan.csv"))
    parser.add_argument("--fin_entrenamiento", default="2012-12-31")
    parser.add_argument("--muestra", type=float, default=None)
    args = parser.parse_args()

    out_res = DIR_ACTUAL / "resultados"
    out_fig = DIR_ACTUAL / "figuras"
    out_res.mkdir(parents=True, exist_ok=True)
    out_fig.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo datos crudos desde {args.datos} (ventana <= {args.fin_entrenamiento}) ...")
    df = cargar_entrenamiento_crudo(Path(args.datos), args.fin_entrenamiento, args.muestra)
    print(f"Créditos de entrenamiento con desenlace conocido: {len(df):,} | Tasa de malos: {df['malo'].mean()*100:.2f}%")

    diagnosticar_datos_para_preprocesamiento(df, out_res)
    resumen = {
        "n_entrenamiento": int(len(df)),
        "tasa_malos_base_pct": round(float(df["malo"].mean() * 100), 2),
        "H1": evaluar_h1_grado_y_tasa(df, out_res, out_fig),
        "H2": evaluar_h2_capacidad_pago(df, out_res, out_fig),
        "H3": evaluar_h3_liquidez_y_buro(df, out_res, out_fig),
        "H4": evaluar_h4_plazo_y_proposito(df, out_res, out_fig),
        "H5": evaluar_h5_estabilidad_y_verificacion(df, out_res, out_fig),
    }

    with open(out_res / "resumen_hipotesis.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print("\n=== RESUMEN DE HIPÓTESIS ===")
    for k in ["H1", "H2", "H3", "H4", "H5"]:
        print(f"  {resumen[k]['hipotesis']} -> {resumen[k]['veredicto']}")


if __name__ == "__main__":
    main()
