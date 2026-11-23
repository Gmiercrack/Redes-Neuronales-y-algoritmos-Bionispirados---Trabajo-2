"""Genera gráficas editoriales de auditoría de datos y partición temporal.

Lee directamente ``data/raw/loan.csv`` y produce cuatro figuras PNG con la
paleta "Lino y grafito" en ``docs/assets/img/figuras/particion/`` para
justificar en el reporte web la partición cronológica (2007-2012 / 2013 /
2014) y la exclusión de 2015.

Uso (desde la raíz del repositorio):
    python scripts/graficas_particion.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# Paleta "Lino y grafito" del reporte web
LAMINA = "#FBFAF7"
PAPEL = "#F1EEE8"
TINTA = "#2A2A2A"
TENUE = "#65635E"
FILETE = "#D6D2C8"
ACENTO = "#2F5D50"
PD_COLOR = "#3A5A80"
LGD_COLOR = "#A24C24"
EL_COLOR = "#6A4C7A"
OK_COLOR = "#2E6B3F"
AVISO_COLOR = "#B57614"
ABIERTO_COLOR = "#C8C3B8"

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CSV = RAIZ / "data" / "raw" / "loan.csv"
SALIDA_DIR = RAIZ / "docs" / "assets" / "img" / "figuras" / "particion"

BUENOS = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}
CASTIGADOS = {
    "Charged Off",
    "Does not meet the credit policy. Status:Charged Off",
}
MORA_SIN_CASTIGO = {
    "Default",
    "Late (31-120 days)",
}
MALOS = CASTIGADOS | MORA_SIN_CASTIGO
ABIERTOS = {
    "Current",
    "Issued",
    "In Grace Period",
    "Late (16-30 days)",
}


def configurar_estilo() -> None:
    """Aplica estilo tipográfico y de color consistente con Lino y grafito."""
    plt.rcParams.update(
        {
            "figure.facecolor": LAMINA,
            "axes.facecolor": LAMINA,
            "savefig.facecolor": LAMINA,
            "text.color": TINTA,
            "axes.labelcolor": TINTA,
            "axes.edgecolor": FILETE,
            "axes.linewidth": 0.9,
            "xtick.color": TENUE,
            "ytick.color": TENUE,
            "font.family": "sans-serif",
            "font.sans-serif": ["IBM Plex Sans", "Segoe UI", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": FILETE,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": True,
            "legend.facecolor": LAMINA,
            "legend.edgecolor": FILETE,
            "legend.fontsize": 8.5,
        }
    )


def cargar_datos() -> pd.DataFrame:
    """Carga únicamente las columnas necesarias de loan.csv."""
    cols = [
        "id",
        "issue_d",
        "term",
        "loan_status",
        "funded_amnt",
        "total_rec_prncp",
        "recoveries",
    ]
    df = pd.read_csv(RUTA_CSV, usecols=cols, low_memory=False)
    df["dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["anio"] = df["dt"].dt.year
    df["loan_status"] = df["loan_status"].astype(str).str.strip()
    df["plazo"] = df["term"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    df["grupo"] = np.select(
        [
            df["loan_status"].isin(BUENOS),
            df["loan_status"].isin(CASTIGADOS),
            df["loan_status"].isin(MORA_SIN_CASTIGO),
            df["loan_status"].isin(ABIERTOS),
        ],
        ["Pagados (0)", "Castigados (PD=1, LGD)", "Mora/Default (PD=1)", "Abiertos (NA)"],
        default="Otros",
    )
    return df


def guardar(fig: plt.Figure, nombre: str, dpi: int = 160) -> None:
    ruta = SALIDA_DIR / nombre
    fig.savefig(ruta, dpi=dpi, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    from PIL import Image

    with Image.open(ruta) as im:
        w, h = im.size
    print(f"Generada: {ruta.relative_to(RAIZ)} ({w}x{h} px, {ruta.stat().st_size:,} bytes)")


def fig_01_evolucion_estados_volumen(df: pd.DataFrame) -> None:
    """Composición anual por estado y volumen originado con cortes temporales."""
    orden = ["Pagados (0)", "Castigados (PD=1, LGD)", "Mora/Default (PD=1)", "Abiertos (NA)"]
    colores = [OK_COLOR, LGD_COLOR, AVISO_COLOR, ABIERTO_COLOR]

    ct = pd.crosstab(df["anio"], df["grupo"])[orden]
    anios = ct.index.to_numpy()
    totales = ct.sum(axis=1)
    pct = ct.div(totales, axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})

    # Panel izquierdo: barras apiladas al 100%
    base = np.zeros(len(anios))
    x = np.arange(len(anios))
    for col, color in zip(orden, colores):
        vals = pct[col].to_numpy()
        ax1.bar(x, vals, bottom=base, color=color, width=0.72, label=col, edgecolor=LAMINA, linewidth=0.5)
        for xi, b, v in zip(x, base, vals):
            if v >= 10:
                txt_col = LAMINA if color != ABIERTO_COLOR else TINTA
                ax1.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=7.5, color=txt_col, fontweight="bold")
        base += vals

    ax1.set_xticks(x, [str(a) for a in anios])
    ax1.set_ylim(0, 102)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax1.set_title("Composición de desenlaces por año de originación")
    ax1.set_ylabel("Porcentaje de créditos originados")
    ax1.grid(axis="x", visible=False)
    ax1.axvline(5.5, color=TINTA, linestyle=":", linewidth=1.1)
    ax1.axvline(7.5, color=LGD_COLOR, linestyle="--", linewidth=1.2)
    ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2, frameon=False)

    # Panel derecho: volumen anual total (cerrados vs abiertos)
    cerrados = ct[["Pagados (0)", "Castigados (PD=1, LGD)", "Mora/Default (PD=1)"]].sum(axis=1) / 1000
    abiertos = ct["Abiertos (NA)"] / 1000

    ax2.axvspan(-0.5, 5.5, color=PD_COLOR, alpha=0.07)
    ax2.axvspan(5.5, 6.5, color=ACENTO, alpha=0.08)
    ax2.axvspan(6.5, 7.5, color=EL_COLOR, alpha=0.08)
    ax2.axvspan(7.5, 8.5, color=LGD_COLOR, alpha=0.10)

    ax2.bar(x, cerrados, color=PD_COLOR, width=0.68, label="Cerrados (PD = 0 o 1)")
    ax2.bar(x, abiertos, bottom=cerrados, color=ABIERTO_COLOR, width=0.68, label="Abiertos / censurados (NA)")

    for xi, c, a in zip(x, cerrados, abiertos):
        tot = c + a
        ax2.text(xi, tot + 7, f"{tot:.0f}k" if tot >= 10 else f"{tot:.1f}k", ha="center", va="bottom", fontsize=7.5, color=TINTA)

    ax2.text(2.5, 430, "Entrenamiento\n2007–2012", ha="center", va="top", fontsize=7.5, color=PD_COLOR, fontweight="bold")
    ax2.text(6.0, 430, "Val\n13", ha="center", va="top", fontsize=7.5, color=ACENTO, fontweight="bold")
    ax2.text(7.0, 430, "Test\n14", ha="center", va="top", fontsize=7.5, color=EL_COLOR, fontweight="bold")
    ax2.text(8.0, 430, "Excl.\n15", ha="center", va="top", fontsize=7.5, color=LGD_COLOR, fontweight="bold")

    ax2.set_xticks(x, [str(a) for a in anios])
    ax2.set_ylim(0, 465)
    ax2.set_title("Volumen originado (miles) y zonas de partición")
    ax2.set_ylabel("Créditos (miles)")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=1, frameon=False)

    fig.tight_layout()
    guardar(fig, "01_evolucion_estados_volumen.png")


def fig_02_maduracion_plazos_36_60(df: pd.DataFrame) -> None:
    """Calendario contractual (36 vs 60 meses) vs. % real de créditos abiertos."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.5), gridspec_kw={"width_ratios": [1.05, 1]})

    anios = list(range(2007, 2016))
    y_pos = np.arange(len(anios))

    # Panel izquierdo: barras tipo Gantt mostrando ventana contractual hasta dic-2015
    for idx, anio in enumerate(anios):
        # Barra de 36 meses (3 años)
        ax1.barh(idx - 0.17, 3.0, left=anio + 0.5, height=0.28, color=PD_COLOR, alpha=0.88,
                 label="Contrato 36 meses (3 años)" if idx == 0 else "")
        # Barra de 60 meses (desde 2010)
        if anio >= 2010:
            ax1.barh(idx + 0.17, 5.0, left=anio + 0.5, height=0.28, color=LGD_COLOR, alpha=0.85,
                     label="Contrato 60 meses (5 años)" if anio == 2010 else "")

    ax1.axvline(2016.0, color=TINTA, linestyle="--", linewidth=1.4)
    ax1.axvspan(2016.0, 2021.3, color=PAPEL, alpha=0.85)
    ax1.text(2016.15, 0.3, "Corte del archivo\nDic 2015\n(sin observación\nposterior)", fontsize=8, color=LGD_COLOR, fontweight="bold", va="top")

    ax1.set_yticks(y_pos, [str(a) for a in anios])
    ax1.invert_yaxis()
    ax1.set_xlim(2007, 2021.3)
    ax1.set_xlabel("Año calendario de vencimiento de cuotas")
    ax1.set_title("Ventana contractual frente al corte de dic-2015")
    ax1.grid(axis="y", visible=False)
    ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=1, frameon=False)

    # Panel derecho: % real de créditos abiertos por año y plazo
    df["es_abierto"] = df["grupo"] == "Abiertos (NA)"
    grp = df.groupby(["anio", "plazo"])["es_abierto"].mean().unstack() * 100

    ax2.plot(grp.index, grp[36], marker="o", linewidth=2.2, color=PD_COLOR, label="Plazo 36 meses")
    s60 = grp[60].dropna()
    ax2.plot(s60.index, s60.values, marker="s", linewidth=2.2, color=LGD_COLOR, label="Plazo 60 meses")

    for a, v in grp[36].items():
        if a in (2010, 2012, 2013, 2014, 2015):
            if a == 2010:
                xy = (-8, 7); va = "bottom"
            elif a == 2012:
                xy = (0, -11); va = "top"
            elif a == 2015:
                xy = (6, -13); va = "top"
            else:
                xy = (8, -12); va = "top"
            ax2.annotate(f"{v:.1f}%", (a, v), textcoords="offset points", xytext=xy,
                         ha="center", va=va, fontsize=7.8, color=PD_COLOR, fontweight="bold")
    for a, v in s60.items():
        if a in (2011, 2012, 2013, 2014, 2015):
            xy = (-8, 7) if a in (2013, 2014, 2015) else (0, 6)
            ax2.annotate(f"{v:.1f}%", (a, v), textcoords="offset points", xytext=xy,
                         ha="center", va="bottom", fontsize=7.8, color=LGD_COLOR, fontweight="bold")

    ax2.axvspan(2014.55, 2015.45, color=LGD_COLOR, alpha=0.10)
    ax2.text(2015.0, 50, "2015\n92,7%\nabierto", ha="center", va="center", fontsize=8, color=LGD_COLOR, fontweight="bold")

    ax2.set_xticks(anios, [str(a) for a in anios])
    ax2.set_ylim(-5, 108)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax2.set_xlabel("Año de originación")
    ax2.set_ylabel("Fracción de créditos abiertos (NA)")
    ax2.set_title("Censura real observada por año y plazo")
    ax2.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=2, frameon=False)

    fig.tight_layout()
    guardar(fig, "02_maduracion_plazos_36_60.png")


def fig_03_riesgo_y_recuperaciones(df: pd.DataFrame) -> None:
    """Evolución anual de tasa de malos (PD) y madurez de recuperaciones post-castigo (LGD)."""
    anios = sorted(df["anio"].unique())

    cerr = df[df["loan_status"].isin(BUENOS | MALOS)].copy()
    cerr["es_malo"] = cerr["loan_status"].isin(MALOS)
    tasa_malos = cerr.groupby("anio")["es_malo"].mean() * 100

    cast = df[df["loan_status"].isin(CASTIGADOS)].copy()
    cast["ead"] = cast["funded_amnt"] - cast["total_rec_prncp"]
    cast = cast[cast["ead"] > 0]
    cast["tiene_rec"] = cast["recoveries"] > 0
    pct_rec = cast.groupby("anio")["tiene_rec"].mean() * 100

    fig, ax = plt.subplots(figsize=(10.6, 4.3))
    ax.axvspan(2006.6, 2012.5, color=PD_COLOR, alpha=0.06, label="Entrenamiento (2007–2012)")
    ax.axvspan(2012.5, 2013.5, color=ACENTO, alpha=0.08, label="Validación (2013)")
    ax.axvspan(2013.5, 2014.5, color=EL_COLOR, alpha=0.08, label="Prueba (2014)")
    ax.axvspan(2014.5, 2015.4, color=LGD_COLOR, alpha=0.12, label="Excluido (2015)")

    ax.plot(tasa_malos.index, tasa_malos.values, marker="o", linewidth=2.3, color=PD_COLOR,
            label="PD: % malos entre cerrados")
    ax.plot(pct_rec.index, pct_rec.values, marker="s", linewidth=2.3, color=LGD_COLOR,
            label="LGD: % castigados con recuperación > 0")

    for a, v in tasa_malos.items():
        dy, va = (8, "bottom") if a == 2015 else (-13, "top")
        ax.annotate(f"{v:.1f}%", (a, v), textcoords="offset points", xytext=(0, dy),
                    ha="center", va=va, fontsize=8.2, color=PD_COLOR, fontweight="bold")
    for a, v in pct_rec.items():
        dy, va = (-14, "top") if a == 2015 else (7, "bottom")
        ax.annotate(f"{v:.1f}%", (a, v), textcoords="offset points", xytext=(0, dy),
                    ha="center", va=va, fontsize=8.2, color=LGD_COLOR, fontweight="bold")

    ax.annotate(
        "Cobranza inconclusa en 2015:\nsolo 11,0% con recuperación\n(89% queda con LGD = 1,00)",
        xy=(2014.95, pct_rec.loc[2015] + 1.5),
        xytext=(2014.0, 68),
        arrowprops={"arrowstyle": "->", "color": LGD_COLOR, "lw": 1.2, "connectionstyle": "arc3,rad=-0.18"},
        fontsize=8.2,
        color=LGD_COLOR,
        fontweight="bold",
        ha="center",
        bbox={"boxstyle": "round,pad=0.32", "fc": LAMINA, "ec": LGD_COLOR, "lw": 0.8, "alpha": 0.95},
    )

    ax.set_xticks(anios, [str(a) for a in anios])
    ax.set_xlim(2006.6, 2015.4)
    ax.set_ylim(0, 92)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylabel("Porcentaje (%)")
    ax.set_xlabel("Año de originación")
    ax.set_title("Evolución anual de la tasa de incumplimiento (PD) y recuperaciones post-castigo (LGD)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, frameon=False)

    fig.tight_layout()
    guardar(fig, "03_riesgo_y_recuperaciones.png")


def fig_04_balance_decision_particion(df: pd.DataFrame) -> None:
    """Balance de volúmenes en la partición adoptada vs. alternativa descartada."""
    cerr = df[df["loan_status"].isin(BUENOS | MALOS)].copy()
    cerr["cast"] = cerr["loan_status"].isin(CASTIGADOS)

    def contar(c1: str, c2: str, c3: str) -> tuple[list[int], list[int]]:
        t1, t2, t3 = pd.Timestamp(c1), pd.Timestamp(c2), pd.Timestamp(c3)
        ms = [
            cerr["dt"] <= t1,
            (cerr["dt"] > t1) & (cerr["dt"] <= t2),
            (cerr["dt"] > t2) & (cerr["dt"] <= t3),
        ]
        pd_cnt = [int(m.sum()) for m in ms]
        lgd_cnt = [int(cerr.loc[m, "cast"].sum()) for m in ms]
        return pd_cnt, lgd_cnt

    pd_act, lgd_act = contar("2012-12-31", "2013-12-31", "2014-12-31")
    pd_alt, lgd_alt = contar("2011-12-31", "2012-12-31", "2013-12-31")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3))
    etqs = ["Entrenamiento", "Validación", "Prueba"]
    x = np.arange(3)
    w = 0.36

    # Panel izquierdo: PD (miles de créditos cerrados)
    b1 = ax1.bar(x - w / 2, [v / 1000 for v in pd_act], width=w, color=PD_COLOR,
                 label="Adoptada: 07–12 / 13 / 14")
    b2 = ax1.bar(x + w / 2, [v / 1000 for v in pd_alt], width=w, color=ABIERTO_COLOR, edgecolor=TENUE,
                 label="Alternativa descartada: 07–11 / 12 / 13")
    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}k",
                 ha="center", va="bottom", fontsize=8, color=PD_COLOR, fontweight="bold")
    for bar in b2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height():.1f}k",
                 ha="center", va="bottom", fontsize=8, color=TENUE)

    ax1.set_xticks(x, etqs)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Créditos cerrados (miles)")
    ax1.set_title("Parte 1 (PD): tamaño de cada conjunto")
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=1, frameon=False)

    # Panel derecho: LGD (miles de créditos castigados)
    b3 = ax2.bar(x - w / 2, [v / 1000 for v in lgd_act], width=w, color=LGD_COLOR,
                 label="Adoptada: 07–12 / 13 / 14 (~1:1:1)")
    b4 = ax2.bar(x + w / 2, [v / 1000 for v in lgd_alt], width=w, color=ABIERTO_COLOR, edgecolor=TENUE,
                 label="Alternativa: Train (6,3k) < Test (14,8k)")
    for bar in b3:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.1f}k",
                 ha="center", va="bottom", fontsize=8, color=LGD_COLOR, fontweight="bold")
    for bar in b4:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{bar.get_height():.1f}k",
                 ha="center", va="bottom", fontsize=8, color=TENUE)

    ax2.set_xticks(x, etqs)
    ax2.set_ylim(0, 18.2)
    ax2.set_ylabel("Créditos castigados (miles)")
    ax2.set_title("Parte 2 (LGD): balance de castigados por conjunto")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=1, frameon=False)

    fig.tight_layout()
    guardar(fig, "04_balance_decision_particion.png")


def main() -> None:
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    configurar_estilo()
    print(f"Cargando {RUTA_CSV} ...")
    df = cargar_datos()
    fig_01_evolucion_estados_volumen(df)
    fig_02_maduracion_plazos_36_60(df)
    fig_03_riesgo_y_recuperaciones(df)
    fig_04_balance_decision_particion(df)


if __name__ == "__main__":
    main()
