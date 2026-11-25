# Trabajo 2 — Redes neuronales para datos tabulares (¿me prestas?)

**Curso:** Redes Neuronales y Algoritmos Bioinspirados · Semestre 2026-02  
**Institución:** Universidad Nacional de Colombia, Sede Medellín — Facultad de Minas  
**Docente:** Juan David Ospina Arango  
**Autores:** Gian Carlo Mier Estrada (Parte 1 · PD, Scorecard y App) · Nestor Enrique Mendoza Rueda (Parte 2 · LGD y Pérdida Esperada)

- **Reporte técnico interactivo (GitHub Pages):** https://gmiercrack.github.io/Redes-Neuronales-y-algoritmos-Bionispirados---Trabajo-2/
- **Simulador interactivo de Scorecard (GitHub Pages):** https://gmiercrack.github.io/Redes-Neuronales-y-algoritmos-Bionispirados---Trabajo-2/app/
- **Video promocional (preliminar):** https://drive.google.com/file/d/17uh0zZLba5SqkAg16Qf_esht64eKBgSp/view?usp=drive_link
- **Video de contribución individual — Gian Carlo Mier Estrada:** https://drive.google.com/file/d/1KxWnywIRGnMsmuaJr9PvH5rYPT540R8r/view?usp=drive_link *(a la espera del video de contribución de Nestor Enrique Mendoza Rueda)*
- **Repositorio:** https://github.com/Gmiercrack/Redes-Neuronales-y-algoritmos-Bionispirados---Trabajo-2

---

## Resumen del problema y resultados finales

Estimamos el riesgo de crédito de consumo sin garantía de **Lending Club** al momento de la aprobación del préstamo combinando tres componentes:

$$
E[L]_i = PD_i \cdot LGD_i \cdot EAD_i
$$

- **Parte 1 (PD):** probabilidad de incumplimiento (clasificación binaria).
- **Parte 2 (LGD):** pérdida dado incumplimiento sobre créditos castigados (regresión acotada en $[0, 1]$).
- **Integración ($E[L]$):** pérdida esperada del portafolio de prueba de 2014 ($n = 74\,154$ créditos; monto desembolsado US$ $1\,061\,569\,725$).

### Resumen de métricas en el conjunto de prueba (2014)

| Tarea | Modelo | Parámetros | Presupuesto / Semillas | Resultado en prueba (2014) |
|---|---|---:|---|---|
| **PD (Parte 1)** | Regresión logística ($L_2, C=0{,}01$) | 55 | 7 config. (rejilla $C$) · determinista | $\text{AUC} = 0{,}6820$ · $\text{Gini} = 0{,}3640$ · $\text{KS} = 26{,}51\%$ · $\text{Brier} = 0{,}1831$ |
| **PD (Parte 1)** | MLP (`54–64–1`, ReLU, SGD) | 3 585 | 30 config. aleatorias · 30 semillas | $\text{AUC} = 0{,}6875 \pm 0{,}0025$ (semilla 17: $0{,}6902$ · $\text{KS} = 27{,}73\%$ · $\text{Brier} = 0{,}1792$) |
| **Scorecard** | $P_0 = 600$, $\theta_0 = 19{:}1$, $\text{PDO} = 20$ | — | Escala log-odds ($A=515{,}04,\; B=28{,}85$) | Monotonía estricta en 10 deciles: D1 ($50{,}39\%$) a D10 ($7{,}58\%$), razón $6{,}65\times$ |
| **LGD (Parte 2)** | Regresión lineal (OLS recortada) | 55 | 1 config. · solución cerrada | $\text{RMSE} = 0{,}0858$ · $\text{MAE} = 0{,}0758$ |
| **LGD (Parte 2)** | MLP (`54–64–32–16–1`, tanh, Adam) | 6 145 | 40 config. aleatorias · 30 semillas | $\text{RMSE} = 0{,}0855 \pm 0{,}0005$ · $\text{MAE} = 0{,}0762 \pm 0{,}0005$ |
| **Pérdida esperada** | `mlp_final` $\times$ `red_lgd` $\times$ `EAD` | — | Fracción pendiente por plazo (36m: $0{,}601$, 60m: $0{,}749$) | **$E[L]_{\text{total}} = \text{US\$ } 154\,013\,735$** ($14{,}51\%$ del monto; rango obs.: US$ $168{,}2\text{M}$–$229{,}5\text{M}$) |

---

## Estructura del repositorio

```text
.
├── README.md
├── requirements.txt                         # dependencias con versiones fijas (Python 3.12)
├── .gitignore                               # excluye data/raw/, .venv/ y temporales pesados
├── data/
│   ├── LCDataDictionary.xlsx                # diccionario oficial de variables (hoja LoanStats)
│   └── raw/loan.csv                         # NO se sube a Git (442 MB); descargable de Kaggle
├── Analisis_del_problema/
│   ├── explicacion_problema.md              # formulación conceptual y auditoría de variables
│   ├── analisis_particion/
│   │   ├── auditoria_particion.py           # auditoría empírica de censura y maduración (2007–2015)
│   │   └── analisis_y_auditoria_particion.md
│   └── analisis_descriptivo/
│       ├── descriptivo.py                   # diagnóstico crudo y contraste visual de hipótesis H1–H5
│       ├── hipotesis_riesgo_resultados.md
│       ├── figuras/                         # figuras H1–H5
│       └── resultados/                      # resumen_hipotesis.json y tablas CSV
├── scripts/
│   └── graficas_particion.py                # genera las 4 figuras editoriales de partición en docs/
├── parte_1/                                 # Probabilidad de Incumplimiento (PD), Scorecard e Interpretabilidad
│   ├── configs/pd.yaml                      # configuración unificada de la Parte 1
│   ├── data/processed/                      # pd_{entrenamiento,validacion,prueba}.parquet + preprocesador
│   ├── src/
│   │   ├── preprocesamiento.py              # aplica el preprocesador compartido de 54 columnas
│   │   ├── regresion_logistica.py           # rejilla explícita de regularización L2 (C)
│   │   ├── red_neuronal.py                  # MLP en PyTorch: --modo entrenar | busqueda | semillas
│   │   ├── evaluacion.py                    # ROC, AUC, Gini, KS, Brier, Log-Loss, ECE y calibración
│   │   ├── scorecard.py                     # transformación P0/odds/PDO, bandas y puntos aditivos
│   │   ├── interpretabilidad.py             # importancia por permutación agrupada (21 variables)
│   │   └── exportar_app.py                  # exporta docs/app/data/modelo_web.{json,js} y valida paridad
│   ├── results/                             # métricas, modelos guardados, semillas, scorecard y permutación
│   └── figures/                             # 7 gráficas de Parte 1
├── parte_2/                                 # Pérdida Dado Incumplimiento (LGD) y Pérdida Esperada (E[L])
│   ├── configs/lgd.yaml                     # configuración unificada de la Parte 2
│   ├── data/processed/                      # lgd_{entrenamiento,validacion,prueba}.parquet + preprocesador
│   ├── src/
│   │   ├── preprocesamiento.py              # ajusta y guarda el preprocesador compartido de 54 columnas
│   │   ├── referencia_lgd.py                # regresión lineal de referencia
│   │   ├── red_lgd.py                       # MLP fraccional en PyTorch: --modo entrenar | busqueda | semillas
│   │   ├── pd_provisional.py                # utilidad histórica (ya sustituida por PD final de Parte 1)
│   │   └── perdida_esperada.py              # integra PD × LGD × EAD sobre la prueba 2014
│   ├── results/                             # métricas, modelos guardados y pérdida esperada definitiva
│   └── figures/                             # figuras de LGD y calibración de pérdida esperada
├── intercambio/
│   ├── LEEME.md                             # contrato de intercambio entre Parte 1 y Parte 2
│   └── pd_prueba.csv                        # PD definitiva del MLP final (semilla 17) sobre prueba 2014
└── docs/                                    # Sitio estático listo para GitHub Pages (/docs)
    ├── .nojekyll                            # desactiva Jekyll para despliegue estático directo
    ├── index.html                           # reporte técnico completo (HTML + CSS + JS + KaTeX)
    ├── assets/                              # estilos, scripts de numeración/citas y todas las figuras PNG
    └── app/                                 # simulador interactivo de Scorecard (HTML/CSS/JS puro)
        ├── index.html
        ├── app.css
        ├── app.js
        └── data/                            # modelo_web.json + modelo_web.js (paridad < 10^-5)
```

---

## Cómo reproducir el proyecto

Todo el proyecto corre en un único entorno virtual de **Python 3.12** con versiones fijas en `requirements.txt`.

### 0. Instalación del entorno (desde la raíz)

```bash
python -m pip install -r requirements.txt
```

### Ruta rápida desde el repositorio clonado (sin descargar `loan.csv` de 442 MB)

Los archivos preprocesados (`parte_1/data/processed/pd_*.parquet`, `parte_2/data/processed/lgd_*.parquet`), el preprocesador compartido (`preprocesador.joblib`), `intercambio/pd_prueba.csv` y los modelos entrenados ya están versionados en Git, por lo que puedes ejecutar directamente los modelos, la evaluación, la scorecard, la exportación web o el servidor local:

```bash
# 1. Parte 2: LGD (referencia lineal y MLP)
cd parte_2
python -m src.referencia_lgd      --config configs/lgd.yaml
python -m src.red_lgd             --config configs/lgd.yaml --modo busqueda
python -m src.red_lgd             --config configs/lgd.yaml --modo semillas

# 2. Parte 1: PD, evaluación, scorecard e interpretabilidad
cd ../parte_1
python -m src.regresion_logistica --config configs/pd.yaml
python -m src.red_neuronal        --config configs/pd.yaml --modo busqueda
python -m src.red_neuronal        --config configs/pd.yaml --modo semillas
python -m src.evaluacion          --config configs/pd.yaml
python -m src.scorecard           --config configs/pd.yaml
python -m src.interpretabilidad   --config configs/pd.yaml

# 3. Integración E[L] = PD × LGD × EAD y actualización del aplicativo web
cd ../parte_2
python -m src.perdida_esperada    --config configs/lgd.yaml
cd ../parte_1
python -m src.exportar_app        --config configs/pd.yaml

# 4. Levantar reporte técnico y aplicativo web localmente (un solo comando desde la raíz)
cd ..
python -m http.server 8000 --directory docs
```

Luego abre en el navegador:
- Reporte técnico: `http://localhost:8000/`
- Simulador de Scorecard: `http://localhost:8000/app/`

*(El simulador también funciona abriendo directamente `docs/app/index.html` vía `file://` gracias a la carga dual `modelo_web.js` + `modelo_web.json`.)*

### Ruta completa desde cero (con datos crudos de Kaggle)

Si deseas regenerar el preanálisis y los archivos `.parquet` desde el archivo original:
1. Descarga `loan.csv` del conjunto [Credit Risk Analysis](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset) y guárdalo en `data/raw/loan.csv`.
2. Ejecuta desde la raíz del repositorio antes de los pasos de modelamiento:

```bash
python Analisis_del_problema/analisis_particion/auditoria_particion.py
python scripts/graficas_particion.py
python Analisis_del_problema/analisis_descriptivo/descriptivo.py
cd parte_2
python -m src.preprocesamiento --config configs/lgd.yaml
cd ../parte_1
python -m src.preprocesamiento --config configs/pd.yaml
```

### Demostración en vivo (`--demo` y `--set`)

- **Modo demostración (< 2 minutos):** añade `--demo` a cualquier comando de entrenamiento, búsqueda, semillas, evaluación, scorecard o interpretabilidad sin sobrescribir los artefactos finales de producción.
- **Modificación en vivo sin editar código ni YAML (`--set`):**

```bash
# Cambiar parámetros de la Scorecard por línea de comandos (desde parte_1/)
python -m src.scorecard --config configs/pd.yaml --set scorecard.pdo=40 --set scorecard.puntaje_base=700

# Entrenar una corrida individual del MLP con otros hiperparámetros (desde parte_1/)
python -m src.red_neuronal --config configs/pd.yaml --modo entrenar --demo --set red_neuronal.entrenar.dropout=0.25

# Modificar parámetros de la red de LGD (desde parte_2/)
python -m src.red_lgd --config configs/lgd.yaml --modo entrenar --demo --set red.entrenar.tasa_aprendizaje=0.005
```

---

## Decisiones metodológicas compartidas

- **Caso de uso antifuga:** estimación al momento de aprobar el crédito (`issue_d`). Solo ingresan **21 variables de entrada** conocidas en la solicitud/aprobación (13 numéricas originales, 3 derivadas y 5 categóricas) transformadas en **54 características** mediante el mismo `preprocesador.joblib` ajustado exclusivamente sobre entrenamiento (2007–2012, $n = 90\,347$). Las **14 columnas post-desembolso** (`prohibidas` en `pd.yaml` y `lgd.yaml`) están bloqueadas por verificación automática.
- **Codificación de desenlace (`loan_status`):**
  - `0` (Bueno): `Fully Paid`, `Does not meet the credit policy. Status:Fully Paid`.
  - `1` (Malo en PD): `Charged Off`, `Default`, `Late (31-120 days)`, `Does not meet the credit policy. Status:Charged Off`.
  - `LGD`: calculado únicamente sobre los **46 009 castigados** (`Charged Off` y su variante de política) como $\text{LGD} = 1 - (\texttt{recoveries} - \texttt{collection\_recovery\_fee}) / (\texttt{funded\_amnt} - \texttt{total\_rec\_prncp})$, acotado a $[0, 1]$.
  - `NA` (Excluidos por desenlace aún abierto): `Current`, `Issued`, `In Grace Period`, `Late (16-30 days)`.
- **Partición cronológica por `issue_d`:** Entrenamiento `2007–2012` ($90\,347$ PD · $14\,419$ LGD), Validación `2013` ($73\,194$ PD · $14\,836$ LGD) y Prueba `2014` ($74\,154$ PD · $13\,981$ LGD). Se excluye `2015` por censura severa del $93\%$ de créditos todavía vigentes.
- **Intercambio entre partes:** `parte_1/src/red_neuronal.py --modo semillas` escribe las probabilidades definitivas del `mlp_final` (semilla 17) en [`intercambio/pd_prueba.csv`](intercambio/pd_prueba.csv), las cuales consume `parte_2/src/perdida_esperada.py` para obtener $E[L]_{\text{total}} = \text{US\$ } 154\,013\,735$ (`pd_es_provisional: false`).

## Nota sobre Git y GitHub Pages

- `data/raw/` está ignorado en `.gitignore` para no subir `loan.csv` (442 MB, superior al límite de 100 MB de GitHub).
- Todo el reporte y la aplicación viven en `docs/` con rutas relativas y `docs/.nojekyll`, listos para publicarse en **Settings → Pages → Branch: `main` / `/docs`**.
