# Análisis Descriptivo y Contraste de Hipótesis de Riesgo (Parte 1 — PD)

**Universo analizado:** Conjunto de entrenamiento con desenlace conocido (`2007–2012`), leído directamente desde `data/raw/loan.csv` **antes** del preprocesamiento y del entrenamiento de modelos.  
**Tamaño de la muestra:** **90 347 créditos** (`75 618` buenos, `14 729` malos; **tasa base de incumplimiento: 16,30 %**).  
**Script reproducible:** [`descriptivo.py`](./descriptivo.py)

---

## 1. Diagnóstico Previo para el Preprocesamiento (`diagnostico_variables_crudas.csv`)

Al inspeccionar los datos crudos de entrenamiento sin imputar ni escalar se justifican empíricamente todas las decisiones tomadas luego en `parte_1/configs/pd.yaml`:

1. **Asimetría extrema y necesidad de mediana + estandarización:**
   - `annual_inc` presenta una **mediana de \$60 000 USD**, una media de \$68 932 USD y un **máximo de \$7 141 778 USD**.
   - `revol_bal` presenta una **mediana de \$10 392 USD** y un **máximo de \$1 207 359 USD**.
   - Por ello, imputar con la **mediana** (en lugar de la media) y **estandarizar** las variables evita que valores extremos dominen los gradientes.
2. **Exactamente 9 variables numéricas con datos faltantes:**
   - `antiguedad_empleo` (`2 897` faltantes; `3,21 %`), `revol_util` (`135`; `0,15 %`), `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `pub_rec`, `total_acc`, `meses_historial_credito` (`29` cada una; `0,03 %`) y `annual_inc` (`4`; `0,004 %`).
   - Esto explica exactamente por qué el preprocesador genera **9 columnas `missingindicator_*`**. Como veremos en $H_5$, tener el dato faltante en antigüedad laboral no es aleatorio: su tasa de mora salta a **`21,33 %`**.
3. **Categorías infrecuentes ($< 0,5\,\%$):**
   - `home_ownership` tiene 2 categorías raras (`OTHER`, `NONE`) y `purpose` tiene 2 (`educational`, `renewable_energy`), justificando agruparlas en `infrequent_sklearn`.

---

## 2. Formulación y Contraste de las 5 Hipótesis

### $H_1$ — Calificación de Riesgo y Tasa de Interés (`grade`, `int_rate`)
- **Hipótesis:** La probabilidad de incumplimiento crece de forma estrictamente monótona a medida que empeora el grado asignado por Lending Club (`A` $\rightarrow$ `G`) y aumenta la tasa de interés.
- **Evidencia (`h1_grado_y_tasa.csv`):**
  - Grado `A`: **`6,66 %`** ($n = 20\,935$, tasa media $7,48\,\%$) $\rightarrow$ `B`: **`13,12 %`** $\rightarrow$ `C`: **`18,71 %`** $\rightarrow$ `D`: **`24,27 %`** $\rightarrow$ `E`: **`32,41 %`** $\rightarrow$ `F`: **`38,56 %`** $\rightarrow$ `G`: **`40,15 %`** ($n = 680$, tasa media $21,55\,\%$).
  - Por tasa de interés: pasa de **`6,00 %`** (en tasas $< 8\,\%$) a **`39,65 %`** (en tasas $\ge 20\,\%$).
  - Correlación de Spearman (`int_rate` vs. `malo`): $\rho = 0.2118$ ($p < 10^{-16}$); $\chi^2$ para `grade`: $p \approx 0$.
- **Veredicto:** **CONFIRMADA.** Es una relación estrictamente monótona y la señal individual más potente del conjunto.
- **Figura asociada:** [`fig1_h1_grado_y_tasa.png`](./figuras/fig1_h1_grado_y_tasa.png).

---

### $H_2$ — Capacidad de Pago y Carga Financiera (`dti`, `annual_inc`)
- **Hipótesis:** Quienes tienen mayor relación deuda/ingreso (`dti`) y menores ingresos anuales (`annual_inc`) presentan mayor incumplimiento por menor margen ante imprevistos.
- **Evidencia (`h2_dti_e_ingreso.csv`):**
  - `dti`: sube de forma continua desde **`13,55 %`** (cuando `dti < 10 %`, $n = 24\,296$) hasta **`18,87 %`** (`20–25 %`), **`19,71 %`** (`25–30 %`) y **`22,27 %`** (`dti >= 30 %`, $n = 2\,232$). Prueba U de Mann-Whitney: $p = 1.37 \times 10^{-79}$.
  - `annual_inc`: baja de forma continua por quintiles desde **`18,99 %`** en el quintil inferior (`Q1: $2k–$38k`) hasta **`12,30 %`** en el quintil superior (`Q5: $90k–$7.14M`). Mann-Whitney: $p = 2.41 \times 10^{-78}$.
- **Veredicto:** **CONFIRMADA.** Ambas variables separan con alta significancia estadística a buenos y malos pagadores.
- **Figura asociada:** [`fig2_h2_dti_e_ingreso.png`](./figuras/fig2_h2_dti_e_ingreso.png).

---

### $H_3$ — Estrés de Liquidez Rotativa y Alertas del Buró (`revol_util`, `inq_last_6mths`, `delinq_2yrs`, `pub_rec`)
- **Hipótesis:** Una utilización elevada del cupo de tarjetas (`revol_util`), múltiples consultas recientes (`inq_last_6mths`) o incidentes negativos previos (`delinq_2yrs`, `pub_rec`) elevan drásticamente el riesgo de impago.
- **Evidencia (`h3_liquidez_y_buro.csv`):**
  - `revol_util`: se duplica desde **`11,08 %`** (`< 25 %`, $n = 15\,927$) hasta **`22,36 %`** (`>= 90 %`, $n = 6\,959$).
  - `inq_last_6mths`: pasa de **`13,54 %`** (`0` consultas) a **`17,10 %`** (`1`), **`18,89 %`** (`2`) y **`23,26 %`** (`3+` consultas, $\chi^2 \; p = 1.73 \times 10^{-141}$).
  - `delinq_2yrs`: sube de **`16,10 %`** (`0` moras) a **`17,61 %`** (`1`) y **`18,18 %`** (`2+`).
  - `pub_rec`: sube de **`16,02 %`** (`0` registros) a **`23,06 %`** (`1+` registros públicos desfavorables, $\chi^2 \; p = 1.27 \times 10^{-29}$).
- **Veredicto:** **CONFIRMADA.** El agotamiento de liquidez rotativa y la búsqueda urgente de crédito son sensores directos de riesgo.
- **Figura asociada:** [`fig3_h3_liquidez_y_buro.png`](./figuras/fig3_h3_liquidez_y_buro.png).

---

### $H_4$ — Condiciones del Contrato: Plazo y Propósito (`plazo_meses`, `purpose`)
- **Hipótesis:** Los créditos a `60 meses` son sustancialmente más riesgosos que los de `36 meses` (incluso controlando por censura), y destinos empresariales (`small_business`) fallan mucho más que créditos de consumo específicos (`car`, `credit_card`).
- **Evidencia (`h4_plazo_y_proposito.csv`):**
  - Plazo en todo entrenamiento (`2007–2012`): **`13,09 %`** en `36 meses` ($n = 74\,959$) vs. **`31,95 %`** en `60 meses` ($n = 15\,388$).
  - **Control estricto de censura (cohorte `2010`, donde ambos plazos terminaron su ciclo):** **`10,92 %`** en `36 meses` ($n = 9\,156$) vs. **`22,51 %`** en `60 meses` ($n = 3\,372$, $\chi^2 \; p = 2.29 \times 10^{-61}$). Esto demuestra que el mayor riesgo a 60 meses es **real** y no solo un artefacto del corte temporal.
  - Propósitos (`purpose`): el más riesgoso es **`small_business` (`28,02 %`**, $n = 3\,166$), seguido por `renewable_energy` (`21,51 %`) y `educational` (`20,85 %`), frente a **`car` (`11,41 %`)**, `major_purchase` (`11,53 %`), `wedding` (`11,90 %`) y `credit_card` (`12,46 %`, $\chi^2 \; p = 8.67 \times 10^{-144}$).
- **Veredicto:** **CONFIRMADA.**
- **Figura asociada:** [`fig4_h4_plazo_y_proposito.png`](./figuras/fig4_h4_plazo_y_proposito.png).

---

### $H_5$ (Contraste) — Verificación de Ingresos, Antigüedad Laboral y Vivienda
- **Hipótesis intuitiva inicial:** Tener ingresos verificados (`Verified`), `10+ años` de antigüedad laboral y vivienda propia o hipotecada (`OWN`/`MORTGAGE`) debería reducir claramente la tasa de incumplimiento frente a no verificados (`Not Verified`), `< 1 año` o arrendatarios (`RENT`).
- **Evidencia (`h5_verificacion_empleo_vivienda.csv`):**
  1. **Paradoja en `verification_status` (REFUTADA):** `Not Verified` tiene una tasa de **`13,70 %`**, `Source Verified` **`16,41 %`** y `Verified` **`19,42 %`** ($\chi^2 \; p = 9.20 \times 10^{-90}$). **¿Por qué ocurre lo contrario a la intuición?** Porque Lending Club no verifica aleatoriamente: exige verificación documental con mayor frecuencia en montos altos o perfiles que ya percibe como más riesgosos (*sesgo de selección en verificación*).
  2. **Efecto muy débil en `antiguedad_empleo` (REFUTADA en tendencia, hallazgo en faltantes):** `< 1 año` registra **`15,81 %`**, `1–3 años` **`15,27 %`**, `4–6 años` **`15,92 %`** y `10+ años` **`16,93 %`** (no disminuye con la antigüedad). En contraste, el grupo **`Sin dato` (`n = 2 897`) alcanza `21,33 %`**, confirmando que la ausencia de empleo reportado es la verdadera señal de riesgo (y validando el `missingindicator` del preprocesador).
  3. **Tenencia de vivienda `home_ownership` (CONFIRMADA parcialmente):** `MORTGAGE` (`15,40 %`) presenta menor tasa de impago que `RENT` (`17,07 %`, $\chi^2 \; p = 5.32 \times 10^{-10}$), aunque `OWN` queda cerca (`16,78 %`).
- **Veredicto:** **REFUTADA PARCIALMENTE.** La intuición ingenua falla en `verification_status` y en la monotonía de `antiguedad_empleo`, dejando dos aprendizajes de negocio clave para contrastar luego contra los pesos del modelo.
- **Figura asociada:** [`fig5_h5_verificacion_empleo_vivienda.png`](./figuras/fig5_h5_verificacion_empleo_vivienda.png).

---

## 3. Síntesis de Veredictos (para contrastar al final del trabajo)

| Hipótesis | Variables | Resultado Empírico | Veredicto |
|---|---|---|---|
| **$H_1$** | `grade`, `int_rate` | `A: 6,66 %` $\rightarrow$ `G: 40,15 %` ($\rho = 0.2118$) | **CONFIRMADA** |
| **$H_2$** | `dti`, `annual_inc` | `dti <10%: 13,55 %` $\rightarrow$ `>=30%: 22,27 %`; `Q1: 18,99 %` $\rightarrow$ `Q5: 12,30 %` | **CONFIRMADA** |
| **$H_3$** | `revol_util`, `inq_last_6mths`, `delinq_2yrs`, `pub_rec` | `revol_util >=90%: 22,36 %`; `3+ consultas: 23,26 %`; `pub_rec 1+: 23,06 %` | **CONFIRMADA** |
| **$H_4$** | `plazo_meses`, `purpose` | En 2010 maduro: `36m 10,92 %` vs `60m 22,51 %`; `small_business: 28,02 %` | **CONFIRMADA** |
| **$H_5$** | `verification_status`, `antiguedad_empleo`, `home_ownership` | `Verified` (`19,42 %`) > `Not Verified` (`13,70 %`); `10+ años` (`16,93 %`) no mejora `< 1 año` (`15,81 %`), pero `Sin dato` salta a `21,33 %` | **REFUTADA PARCIALMENTE** |
