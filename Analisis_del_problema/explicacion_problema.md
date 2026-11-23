# Entendimiento del Problema, Lógica Crediticia y Guía de Variables

**Proyecto:** Modelo de Riesgo Crediticio y Pérdida Esperada — Lending Club  
**Cátedra:** Redes Neuronales y Algoritmos Bioinspirados (2026-02) — Universidad Nacional de Colombia  
**Propósito:** Este documento detalla la lógica financiera del problema, el caso de uso, el rol de cada variable de entrada y la base conceptual para el diseño y explicabilidad del **Aplicativo Web**.

---

## 1. Contexto del Negocio y Caso de Uso

### 1.1 ¿Qué es Lending Club y qué problema enfrenta?
Lending Club es una plataforma estadounidense de préstamos entre pares (*peer-to-peer lending*). Conecta a personas que necesitan dinero (prestatarios) con inversionistas o bancos que financian dichos préstamos.

**La naturaleza del producto:**
Son créditos de consumo **sin garantía prendaria ni hipotecaria** (*unsecured consumer loans*). A diferencia de un crédito hipotecario (respaldado por un inmueble) o vehicular (respaldado por el vehículo), en Lending Club no existe un activo tangible de respaldo. Si el prestatario entra en cesación de pagos, la entidad enfrenta una pérdida crediticia directa.

### 1.2 Definición del Caso de Uso
- **¿Quién usa el modelo?** El analista de riesgo crediticio o el motor automatizado de originación de la entidad (o un inversionista/banco secundario que evalúa créditos ya pre-tarificados al momento de su emisión).
- **¿En qué momento?** Al momento exacto de la **solicitud y aprobación del crédito**, antes de confirmar el desembolso del dinero.
- **¿Para qué decisiones?**
  1. **Admisión / Aprobación:** Decidir si se aprueba o se rechaza la solicitud de crédito con base en la Probabilidad de Incumplimiento ($PD$).
  2. **Scorecard:** Asignar un puntaje transparente (por ejemplo, de 300 a 850 puntos) comprensible tanto para el banco como para el cliente.

#### ¿Con qué información se entrena y decide? (Entradas permitidas)
De las 74 columnas del conjunto original, únicamente ingresan **21 variables de solicitud y perfil crediticio disponibles en la originación** (que tras imputación con indicadores de faltantes y codificación *one-hot* producen **54 variables numéricas de entrada**):
- **13 variables numéricas originales de solicitud y buró:** monto solicitado (`loan_amnt`), monto comprometido (`funded_amnt`), tasa de interés (`int_rate`), cuota mensual (`installment`), ingreso anual (`annual_inc`), relación deuda/ingreso (`dti`), moras en últimos 2 años (`delinq_2yrs`), consultas en últimos 6 meses (`inq_last_6mths`), cuentas abiertas (`open_acc`), registros públicos adversos (`pub_rec`), saldo rotativo (`revol_bal`), utilización rotativa (`revol_util`) y cuentas totales (`total_acc`).
- **3 variables numéricas derivadas al momento de originación:** plazo en meses (`plazo_meses`), antigüedad laboral (`antiguedad_empleo`) y meses de historial crediticio hasta `issue_d` (`meses_historial_credito`).
- **5 variables categóricas:** grado asignado (`grade`), tenencia de vivienda (`home_ownership`), verificación de ingresos (`verification_status`), propósito del crédito (`purpose`) y estado inicial de listado (`initial_list_status`).
*(Nota de supuesto compartido: se mantienen `grade` e `int_rate` bajo el supuesto de evaluar créditos ya tarificados en la originación).*

#### ¿Qué información NO entra? (Prevención estricta de fuga de información / *Data Leakage*)
Para garantizar que el modelo no "vea el futuro", el pipeline bloquea automáticamente mediante validación (`verificar_sin_fuga`) **14 columnas prohibidas** conocidas únicamente después de desembolsar el crédito (definiciones tomadas de `LCDataDictionary.xlsx`, hoja `LoanStats`):

1. **Estado actual del crédito (variable de desenlace):**
   - `loan_status` (*"Current status of the loan"*): define la variable objetivo, nunca puede ser entrada.
2. **Pagos, saldos y fechas posteriores al desembolso:**
   - `out_prncp` / `out_prncp_inv`: saldo de capital pendiente (*"Remaining outstanding principal..."*).
   - `total_pymnt` / `total_pymnt_inv`: pagos totales recibidos a la fecha (*"Payments received to date..."*).
   - `total_rec_prncp`: capital recibido a la fecha (*"Principal received to date"*).
   - `total_rec_int`: intereses recibidos a la fecha (*"Interest received to date"*).
   - `total_rec_late_fee`: cargos por mora recibidos (*"Late fees received to date"*).
   - `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`: fecha/monto del último pago y próxima fecha programada.
3. **Recuperaciones post-castigo (uso exclusivo como objetivo en Parte 2 - LGD):**
   - `recoveries` (*"post charge off gross recovery"*) y `collection_recovery_fee` (*"post charge off collection fee"*).
4. **Seguimiento posterior del buró:**
   - `last_credit_pull_d` (*"The most recent month LC pulled credit for this loan"*): refleja consultas hechas después de originar el crédito.

Además, se evita fuga temporal aplicando cortes estrictos por `issue_d` (entrenamiento 2007–2012, validación 2013 y prueba 2014) y excluyendo por completo los **618 849** créditos sin desenlace cerrado (`Current`, `Issued`, etc.) y los **30 835** créditos de **2015** con desenlace (por sesgo de censura temprana).

### 1.3 Ficha Técnica de Datos, Origen y Alcance de Decisión

| Aspecto | Detalle Oficial del Proyecto |
|---|---|
| **Fuente de datos** | Kaggle — *Credit Risk Analysis* (`ranadeep/credit-risk-dataset`), créditos de Lending Club |
| **URL del conjunto** | https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset |
| **Fecha de descarga** | **27 de octubre de 2026** |
| **Diccionario citado** | `data/LCDataDictionary.xlsx` (hoja `LoanStats`) |
| **Tamaño crudo (`loan.csv`)** | **887 379 registros** y **74 columnas** |
| **Periodo cubierto en crudo** | Junio de 2007 a diciembre de 2015 (`issue_d`) |
| **Momento de decisión** | En la **originación / aprobación** del crédito (antes del desembolso) |
| **Información disponible en ese momento** | **21 variables base** de solicitud y buró (**54 variables transformadas**); **14 variables post-crédito prohibidas** |
| **Registros con desenlace conocido** | **268 530 créditos** (se excluyen **618 849** abiertos/censurados sin estado final) |
| **Universo efectivo modelado (2007–2014)** | **237 695 créditos** (se excluyen **30 835** de 2015 por censura del 93 % en ese año) |
| **Partición temporal** | **Entrenamiento (2007–2012):** 90 347 (16.30 % malos) · **Validación (2013):** 73 194 (22.95 % malos) · **Prueba (2014):** 74 154 (26.22 % malos) |

---

## 2. Definición de la Variable Objetivo: Probabilidad de Incumplimiento ($PD$)

El problema de la Parte 1 es de **clasificación binaria supervisada**: predecir si un crédito resultará ser "malo" ($Y = 1$) o "bueno" ($Y = 0$).

### 2.1 Codificación a partir de `loan_status`
| Estado Original (`loan_status`) | Código ($Y$) | Justificación Financiera |
|---|:---:|---|
| `Fully Paid` | 0 | Buen pagador: amortizó capital e intereses en su totalidad. |
| `Does not meet the credit policy. Status:Fully Paid` | 0 | Buen pagador confirmado (aunque no cumplía políticas iniciales). |
| `Charged Off` | 1 | Mal pagador: declarado incobrable y castigado contablemente. |
| `Default` | 1 | Mal pagador: impago confirmado y cesación prolongada. |
| `Late (31-120 days)` | 1 | Mal pagador: mora grave; probabilidad de recuperación espontánea mínima. |
| `Does not meet the credit policy. Status:Charged Off` | 1 | Mal pagador sin política inicial que terminó en pérdida. |
| `Current`, `Issued`, `In Grace Period`, `Late (16-30 days)` | **NA (Excluidos)** | Créditos abiertos/censurados; su desenlace final aún no se conoce. |

---

## 3. Anatomía del Riesgo: Variables de Entrada y su Rol

Las variables seleccionadas operan como "sensores" que miden las diferentes dimensiones del riesgo crediticio:

### Dimensión 1: Capacidad de Pago y Carga Financiera (Flujo de Caja)
*Evalúa si los ingresos netos del solicitante son suficientes para atender el nuevo compromiso.*

- **`annual_inc` (Ingreso Anual Declarado):**
  - *Cita oficial (LCDataDictionary):* *"The self-reported annual income provided by the borrower during registration."*
  - *Impacto en riesgo:* A mayor ingreso, mayor capacidad de ahorro y resiliencia ante contingencias económicas.
- **`dti` (*Debt-to-Income* o Relación Deuda/Ingreso):**
  - *Cita oficial (LCDataDictionary):* *"A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income."*
  - *Impacto en riesgo:* **Predictor crítico.** Si el DTI es 45%, casi la mitad de los ingresos mensuales ya están comprometidos en otras deudas. Un DTI elevado indica asfixia financiera y alta vulnerabilidad al impago.
- **`installment` (Cuota Mensual del Préstamo):**
  - *Cita oficial (LCDataDictionary):* *"The monthly payment owed by the borrower if the loan originates."*
  - *Impacto en riesgo:* Carga directa que el préstamo solicitado impondrá sobre el flujo mensual de caja.
- **`loan_amnt` y `funded_amnt` (Monto Solicitado y Comprometido):**
  - *Cita oficial (LCDataDictionary):* `loan_amnt`: *"The listed amount of the loan applied for by the borrower."* / `funded_amnt`: *"The total amount committed to that loan at that point in time."*
  - *Impacto en riesgo:* Montos mayores exigen un esfuerzo prolongado de pago, elevando la exposición del acreedor.

---

### Dimensión 2: Carácter e Historial Crediticio (Comportamiento Pasado)
*Evalúa el hábito y la disciplina de pago del cliente a lo largo de su vida financiera.*

- **`delinq_2yrs` (Moras de 30+ días en los últimos 2 años):**
  - *Cita oficial (LCDataDictionary):* *"The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years."*
  - *Impacto en riesgo:* Un cliente que incurrió recientemente en moras presenta una probabilidad significativamente superior de reincidir en impago.
- **`pub_rec` (Registros Públicos Desfavorables):**
  - *Cita oficial (LCDataDictionary):* *"Number of derogatory public records."*
  - *Impacto en riesgo:* Incluye bancarrotas previas, quiebras legales o embargos judiciales. Señal crítica de insolvencia estructural previa.
- **`inq_last_6mths` (Consultas al Buró en los Últimos 6 Meses):**
  - *Cita oficial (LCDataDictionary):* *"The number of inquiries in past 6 months (excluding auto and mortgage inquiries)."*
  - *Impacto en riesgo:* Múltiples solicitudes de crédito recientes revelan una búsqueda urgente de liquidez (*credit hunger*), síntoma precursor de insolvencia.
- **`meses_historial_credito` (Antigüedad Crediticia, de `earliest_cr_line` a `issue_d`):**
  - *Cita oficial (LCDataDictionary):* `earliest_cr_line`: *"The month the borrower's earliest reported credit line was opened."*
  - *Impacto en riesgo:* Un historial crediticio maduro (ej. > 10 años) con bajo reporte de incidentes demuestra estabilidad y confiabilidad financiera.

---

### Dimensión 3: Estrés de Liquidez y Manejo de Líneas Rotativas
*Evalúa si el cliente vive al límite de su capacidad crediticia.*

- **`revol_util` (Porcentaje de Utilización de Tarjetas de Crédito):**
  - *Cita oficial (LCDataDictionary):* *"Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit."*
  - *Impacto en riesgo:* **Predictor clave.** Un solicitante que usa el 90% de los cupos de sus tarjetas no tiene margen de maniobra ni liquidez ante imprevistos. Tasas de utilización inferiores al 30% son sinónimo de solvencia y disciplina.
- **`revol_bal` (Saldo Total Rotativo Adeudado):**
  - *Cita oficial (LCDataDictionary):* *"Total credit revolving balance."*
  - *Impacto en riesgo:* Magnitud nominal de la deuda corriente a corto plazo.
- **`open_acc` y `total_acc` (Cuentas Abiertas y Cuentas Totales):**
  - *Cita oficial (LCDataDictionary):* `open_acc`: *"The number of open credit lines in the borrower's credit file."* / `total_acc`: *"The total number of credit lines currently in the borrower's credit file."*
  - *Impacto en riesgo:* Miden la profundidad, diversificación y volumen de líneas administradas por el solicitante.

---

### Dimensión 4: Estabilidad Laboral y Arraigo
*Evalúa la certeza de continuidad en la generación de ingresos y el respaldo patrimonial.*

- **`antiguedad_empleo` (Años en el Empleo Actual, derivado de `emp_length`):**
  - *Cita oficial (LCDataDictionary):* *"Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years."*
  - *Impacto en riesgo:* Mayor antigüedad se traduce en estabilidad laboral y menor probabilidad de cese de ingresos por despido intempestivo.
- **`home_ownership` (Tenencia de Vivienda: RENT, OWN, MORTGAGE, OTHER):**
  - *Cita oficial (LCDataDictionary):* *"The home ownership status provided by the borrower during registration."*
  - *Impacto en riesgo:* Los propietarios (`OWN`) o compradores con hipoteca (`MORTGAGE`) tienen mayor arraigo y respaldo patrimonial indirecto que los arrendatarios (`RENT`).
- **`verification_status` (Estado de Verificación de Ingresos):**
  - *Cita oficial (LCDataDictionary):* En el diccionario aparece catalogado conceptualmente como `is_inc_v`: *"Indicates if income was verified by LC, not verified, or if the income source was verified."*
  - *Impacto en riesgo:* Señala la fidelidad documental de los ingresos declarados por el cliente.

---

### Dimensión 5: Características del Crédito y Tarificación
*Parámetros del contrato y evaluación preliminar de riesgo.*

- **`plazo_meses` (Plazo del Préstamo, derivado de `term`: 36 vs 60 meses):**
  - *Cita oficial (LCDataDictionary):* *"The number of payments on the loan. Values are in months and can be either 36 or 60."*
  - *Impacto en riesgo:* A mayor horizonte temporal (5 años vs. 3 años), mayor incertidumbre económica y mayor exposición a eventos de vida desfavorables.
- **`purpose` (Destino del Crédito):**
  - *Cita oficial (LCDataDictionary):* *"A category provided by the borrower for the loan request."*
  - *Impacto en riesgo:* Créditos destinados a `small_business` (emprendimientos) tienen una tasa histórica de quiebra significativamente superior a los destinados a `debt_consolidation` (refinanciación de tarjetas a menor tasa).
- **`grade` e `int_rate` (Grado de Riesgo y Tasa de Interés Asignada):**
  - *Cita oficial (LCDataDictionary):* `grade`: *"LC assigned loan grade"* / `int_rate`: *"Interest Rate on the loan."*
  - *Impacto en riesgo:* Reflejan la prima de riesgo del crédito. Tasas elevadas incrementan la cuota mensual, lo cual puede generar una espiral de impago si el cliente ya está ajustado.
- **`initial_list_status` (Estado Inicial de Listado: `w` o `f`):**
  - *Cita oficial (LCDataDictionary):* *"The initial listing status of the loan. Possible values are – W, F."*
  - *Impacto en riesgo:* Indica cómo salió el crédito al mercado en la plataforma: en bloque completo para inversionistas institucionales (`w`, *whole*) o fraccionado para inversionistas minoristas (`f`, *fractional*). Captura diferencias en el proceso de selección y apetito de riesgo de la plataforma al momento de emitir el préstamo.

---

## 4. Lineamientos para el Diseño del Aplicativo Web

Para dar cumplimiento a los requisitos de la aplicación web exigida en el trabajo (Puntaje de scorecard, PD, percentil poblacional y análisis de sensibilidad *what-if*), este documento define la lógica funcional:

1. **Formulario de Entrada Amigable:**
   - Agrupar los campos en secciones claras: *Datos Financieros*, *Historial y Crédito*, *Condiciones del Préstamo*.
   - Incluir tooltips explicativos con base en este documento para que el usuario entienda qué significa cada término (ej. "¿Qué es DTI?" o "¿Qué es la utilización de crédito?").
2. **Resultados Intuitivos:**
   - Visualizador tipo velocímetro / tacómetro con el **Scorecard (ej. 300 a 850 puntos)** y la **Probabilidad de Incumplimiento ($PD$)** estimada por la Red Neuronal y la Regresión Logística.
   - Gráfico de distribución de la población: Mostrar dónde se ubica el cliente en comparación con el conjunto de prueba (percentil de riesgo).
3. **Módulo de Sensibilidad Interactivo (*What-If Simulator*):**
   - Permitir al usuario mover barras deslizantes (*sliders*) para responder preguntas clave:
     - *"¿Cómo mejoraría mi puntaje si reduzco la utilización de mis tarjetas (`revol_util`) del 85% al 25%?"*
     - *"¿Qué pasa si elijo un plazo de 36 meses en lugar de 60 meses?"*
     - *"¿Cuánto debe aumentar mi ingreso anual para acceder a un puntaje de bajo riesgo?"*

