# Auditoría y Análisis Empírico de la Partición Temporal (`loan.csv`)

**Proyecto:** Modelo de Riesgo Crediticio y Pérdida Esperada — Lending Club  
**Propósito de este documento:** Explicar paso a paso, con los **datos reales extraídos de las 887 379 filas** de `data/raw/loan.csv`, cómo se distribuyen las 10 categorías de `loan_status` año por año (`2007–2015`), por qué la duración contractual de los créditos (`36 meses` vs `60 meses`) explica físicamente en qué años ya existen préstamos pagados por completo frente a préstamos aún en curso, cómo impacta esto a la variable objetivo de **PD** (Parte 1) y de **LGD** (Parte 2), y evaluar críticamente las alternativas de partición que están bajo control del equipo.

---

## 1. Radiografía Real del Dataset: Año de Originación vs. las 10 Categorías de `loan_status`

El archivo `data/raw/loan.csv` contiene **887 379 préstamos** originados entre **junio de 2007 y diciembre de 2015**, fotografiados al cierre de **diciembre de 2015**. La columna `loan_status` toma exactamente **10 valores distintos**:

### 1.1 Conteo exacto de préstamos por año y por estado (`loan_status`)

| Año (`issue_d`) | `Fully Paid` (0) | `No Policy: Fully Paid` (0) | `Charged Off` (1) | `No Policy: Charged Off` (1) | `Default` (1) | `Late (31-120 days)` (1) | `Current` (NA) | `In Grace Period` (NA) | `Late (16-30 days)` (NA) | `Issued` (NA) | **TOTAL** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2007** | 206 | 239 | 45 | 113 | 0 | 0 | 0 | 0 | 0 | 0 | **603** |
| **2008** | 1 315 | 582 | 247 | 249 | 0 | 0 | 0 | 0 | 0 | 0 | **2 393** |
| **2009** | 4 122 | 436 | 594 | 129 | 0 | 0 | 0 | 0 | 0 | 0 | **5 281** |
| **2010** | 10 038 | 731 | 1 483 | 270 | 1 | 5 | 7 | 0 | 2 | 0 | **12 537** |
| **2011** | 16 470 | 0 | 3 205 | 0 | 8 | 46 | 1 953 | 28 | 11 | 0 | **21 721** |
| **2012** | 41 479 | 0 | 8 084 | 0 | 28 | 222 | 3 444 | 83 | 27 | 0 | **53 367** |
| **2013** | 56 396 | 0 | 14 836 | 0 | 182 | 1 780 | 60 264 | 977 | 320 | 0 | **134 755** |
| **2014** | 54 713 | 0 | 13 981 | 0 | 613 | 4 847 | 158 558 | 2 058 | 858 | 0 | **235 628** |
| **2015** | 22 984 | 0 | 2 773 | 0 | 387 | 4 691 | 377 553 | 3 107 | 1 139 | 8 460 | **421 094** |
| **TOTAL** | **207 723** | **1 988** | **45 248** | **761** | **1 219** | **11 591** | **601 779** | **6 253** | **2 357** | **8 460** | **887 379** |

### 1.2 Distribución porcentual dentro de cada año (filas suman 100 %)

| Año (`issue_d`) | `Fully Paid` | `No Policy: Fully Paid` | `Charged Off` | `No Policy: Charged Off` | `Default` | `Late (31-120)` | `Current` | `In Grace` | `Late (16-30)` | `Issued` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2007** | 34,16 % | 39,64 % | 7,46 % | 18,74 % | 0,00 % | 0,00 % | **0,00 %** | 0,00 % | 0,00 % | 0,00 % |
| **2008** | 54,95 % | 24,32 % | 10,32 % | 10,41 % | 0,00 % | 0,00 % | **0,00 %** | 0,00 % | 0,00 % | 0,00 % |
| **2009** | 78,05 % | 8,26 % | 11,25 % | 2,44 % | 0,00 % | 0,00 % | **0,00 %** | 0,00 % | 0,00 % | 0,00 % |
| **2010** | 80,07 % | 5,83 % | 11,83 % | 2,15 % | 0,01 % | 0,04 % | **0,06 %** | 0,00 % | 0,02 % | 0,00 % |
| **2011** | 75,83 % | 0,00 % | 14,76 % | 0,00 % | 0,04 % | 0,21 % | **8,99 %** | 0,13 % | 0,05 % | 0,00 % |
| **2012** | 77,72 % | 0,00 % | 15,15 % | 0,00 % | 0,05 % | 0,42 % | **6,45 %** | 0,16 % | 0,05 % | 0,00 % |
| **2013** | 41,85 % | 0,00 % | 11,01 % | 0,00 % | 0,14 % | 1,32 % | **44,72 %** | 0,73 % | 0,24 % | 0,00 % |
| **2014** | 23,22 % | 0,00 % | 5,93 % | 0,00 % | 0,26 % | 2,06 % | **67,29 %** | 0,87 % | 0,36 % | 0,00 % |
| **2015** | 5,46 % | 0,00 % | 0,66 % | 0,00 % | 0,09 % | 1,11 % | **89,66 %** | 0,74 % | 0,27 % | 2,01 % |

### 1.3 Cuatro hallazgos inmediatos que se ven en la tabla
1. **Las categorías `Does not meet the credit policy...` (`1 988` pagados y `761` castigados) solo existen entre 2007 y 2010 (`0` desde 2011):** eran préstamos originados bajo un programa antiguo de Lending Club que terminó en 2010. Todos ellos ya cerraron su ciclo.
2. **La categoría `Issued` (`8 460` créditos) solo existe en 2015:** son préstamos recién desembolsados a finales de 2015 que aún no habían facturado siquiera su primera cuota.
3. **Los estados `Late (31-120 days)` y `Default` son estados en tránsito hacia `Charged Off`:** en los años antiguos (`2007–2011`) valen prácticamente `0`, porque cualquier préstamo de 2008 o 2010 que estuvo en mora de 31–120 días terminó o poniéndose al día o pasando a `Charged Off` (> 120–150 días de mora) mucho antes de 2015. En cambio, en `2013`, `2014` y `2015` vemos a los clientes que estaban en plena ventana de 31 a 120 días de atraso justo cuando se tomó la foto en diciembre de 2015.
4. **La categoría `Current` (`601 779` créditos al día, el 67,8 % de todo el archivo) es nula en 2007–2009 (`0`), mínima en 2011–2012 (`8,99 %` y `6,45 %`) y mayoritaria en 2013–2015 (`44,72 %`, `67,29 %`, `89,66 %`).** ¿Por qué ocurre exactamente ese salto entre 2010/2012 y 2013? La respuesta está en el plazo contractual (`term`).

---

## 2. La Explicación Física del Calendario: Plazo (`36` vs `60` meses) y Maduración del Crédito

En Lending Club solo existen dos plazos de amortización: **36 meses (3 años)** y **60 meses (5 años)**. Como la base de datos se cortó en **diciembre de 2015**:
- Un préstamo de **36 meses** necesita **3 años calendario** para llegar a su última cuota normal.
- Un préstamo de **60 meses** necesita **5 años calendario** para llegar a su última cuota normal.

Al cruzar **año de originación × plazo (`36` vs `60` meses)** en `data/raw/loan.csv`, obtenemos la prueba exacta:

| Año | Plazo (`term`) | Año en que vence la última cuota | `Bueno (0)` Pagados | `Malo (1)` Impagos | `Abierto (NA)` En curso | **Total originados** | **% Abierto (Censura)** | **% Malos (sobre cerrados)** |
|---|:---:|---|---:|---:|---:|---:|---:|---:|
| **2007** | 36 meses | **2010** *(ya venció)* | 445 | 158 | 0 | 603 | **0,00 %** | 26,20 % |
| **2008** | 36 meses | **2011** *(ya venció)* | 1 897 | 496 | 0 | 2 393 | **0,00 %** | 20,73 % |
| **2009** | 36 meses | **2012** *(ya venció)* | 4 558 | 723 | 0 | 5 281 | **0,00 %** | 13,69 % |
| **2010** | 36 meses | **2013** *(ya venció)* | 8 156 | 1 000 | 0 | 9 156 | **0,00 %** | 10,92 % |
| **2010** | 60 meses | **2015** *(ya venció)* | 2 613 | 759 | 9 | 3 381 | **0,27 %** | 22,51 % |
| **2011** | 36 meses | **2014** *(ya venció)* | 12 602 | 1 499 | 0 | 14 101 | **0,00 %** | 10,63 % |
| **2011** | 60 meses | **2016** *(vence después de dic-2015)* | 3 868 | 1 760 | 1 992 | 7 620 | **26,14 %** | 31,27 % |
| **2012** | 36 meses | **2015** *(ya venció)* | 37 488 | 5 937 | 45 | 43 470 | **0,10 %** | 13,67 % |
| **2012** | 60 meses | **2017** *(vence después de dic-2015)* | 3 991 | 2 397 | 3 509 | 9 897 | **35,46 %** | 37,52 % |
| **2013** | 36 meses | **2016** *(vence después de dic-2015)* | 45 858 | 11 093 | 43 429 | 100 380 | **43,26 %** | 19,48 % |
| **2013** | 60 meses | **2018** *(vence después de dic-2015)* | 10 538 | 5 705 | 18 132 | 34 375 | **52,75 %** | 35,12 % |
| **2014** | 36 meses | **2017** *(vence después de dic-2015)* | 41 886 | 11 904 | 108 779 | 162 569 | **66,91 %** | 22,13 % |
| **2014** | 60 meses | **2019** *(vence después de dic-2015)* | 12 827 | 7 537 | 52 695 | 73 059 | **72,13 %** | 37,01 % |
| **2015** | 36 meses | **2018** *(vence después de dic-2015)* | 16 474 | 4 663 | 262 035 | 283 172 | **92,54 %** | 22,06 % |
| **2015** | 60 meses | **2020** *(vence después de dic-2015)* | 6 510 | 3 188 | 128 224 | 137 922 | **92,97 %** | 32,87 % |
| **TOTAL** | — | — | **209 711** | **58 819** | **618 849** | **887 379** | **69,74 %** | **21,90 %** |

### ¿Cómo se lee esta tabla para entender la separación de los datos?

1. **Por qué `2007–2012` es el único bloque con ciclo natural completo:**
   - Entre **2007 y 2009** solo existían préstamos de 36 meses (`0` de 60 meses). Todos vencieron entre 2010 y 2012 (`0,00 %` abiertos).
   - En **2010** aparecieron los préstamos de 60 meses; tanto los de 36 meses (vencidos en 2013) como los de 60 meses (vencidos en 2015) ya terminaron (`0,00 %` y `0,27 %` abiertos).
   - En **2011 y 2012**, **todos los créditos de 36 meses** (`14 101` en 2011 y `43 470` en 2012) ya cumplieron sus 36 cuotas mensuales antes de diciembre de 2015 (`0,00 %` y `0,10 %` abiertos). Los únicos que siguen abiertos en 2011 y 2012 (`1 992` y `3 509`) son préstamos de **60 meses** cuyas últimas cuotas vencían en 2016 y 2017.
   - **Conclusión:** hasta **2012** observamos tanto a los buenos pagadores que pagaron mes a mes sus 36 cuotas completas como a los que prepagaron y a los que incumplieron.

2. **Qué significa realmente un crédito `Fully Paid` en `2013`, `2014` y `2015` (sesgo de prepago vs. fallo temprano):**
   - Un préstamo de 36 meses otorgado en **2014** tenía su última cuota programada en **2017**. ¿Por qué entonces hay `41 886` préstamos de 36 meses y `12 827` de 60 meses en 2014 con estado `Fully Paid`?
   - Porque son deudores que **prepagaron / cancelaron anticipadamente** toda su deuda en menos de 1 o 2 años.
   - Mientras tanto, los deudores de 2014 que siguen pagando puntualmente su cuota mes a mes siguen en estado **`Current` (`158 558` créditos)** y quedan fuera (`NA`).
   - En cambio, los deudores de 2014 que dejaron de pagar en el primer año **ya cayeron en `Charged Off` o `Late (31-120 days)` (`19 441` créditos)**.
   - **Efecto directo:** al quitar los `158 558` `Current` (cuya gran mayoría son buenos pagadores que simplemente siguen pagando a plazo), la proporción de malos entre los que ya cerraron se infla de **`16,30 %`** (`2007–2012`) a **`22,95 %`** (`2013`) y **`26,22 %`** (`2014`).

---

## 3. El Segundo Efecto del Calendario: ¿Qué pasa con las Recuperaciones (`recoveries`) y la LGD por Año?

Para la **Parte 2 (LGD)**, el problema de calendario es todavía más marcado porque la recuperación (`recoveries`) ocurre **después** de que el crédito entra en mora de 120+ días, se declara `Charged Off` y pasa a agencias de cobranza externas.

Veamos los **46 009 créditos castigados** (`Charged Off` + `Does not meet... Charged Off`) en `data/raw/loan.csv` desglosados por año:

| Año de originación | Créditos castigados (`estados_lgd`) | Castigados con `recoveries > 0` | **% con recuperación registrada** | **LGD media** | **LGD mediana** | **% con `LGD = 1` (pérdida 100 %)** |
|---|---:|---:|---:|---:|---:|---:|
| **2007** | 158 | 100 | **63,29 %** | 0,9207 | 0,9555 | 36,71 % |
| **2008** | 496 | 277 | **55,85 %** | 0,9308 | 0,9771 | 44,15 % |
| **2009** | 723 | 477 | **65,98 %** | 0,9257 | 0,9555 | 34,02 % |
| **2010** | 1 753 | 1 304 | **74,39 %** | 0,9246 | 0,9551 | 25,61 % |
| **2011** | 3 205 | 2 479 | **77,35 %** | 0,9222 | 0,9433 | 22,65 % |
| **2012** | 8 084 | 6 050 | **74,84 %** | 0,9114 | 0,9428 | 25,16 % |
| **2013** | 14 836 | 8 472 | **57,10 %** | 0,9179 | 0,8845 | 42,90 % |
| **2014** | 13 981 | 5 212 | **37,28 %** | 0,9452 | **1,0000** | **62,72 %** |
| **2015** | 2 773 | 306 | **11,03 %** | 0,9807 | **1,0000** | **88,97 %** |
| **Total / Global** | **46 009** | **24 677** | **53,64 %** | **0,9296** | **0,9813** | **46,36 %** |

### Observaciones críticas sobre LGD:
1. **Entre 2010 y 2012**, cerca de **3 de cada 4 créditos castigados (`74,4 %–77,4 %`)** registran recuperaciones posteriores al castigo, y solo ~`23 %–25 %` tienen pérdida total (`LGD = 1`).
2. **En 2013**, la fracción con recuperación baja a **`57,10 %`** (`42,90 %` con `LGD = 1`).
3. **En 2014**, la fracción con recuperación cae a **`37,28 %`**: casi **dos tercios (`62,72 %`)** de los castigados de 2014 aún aparecen con `recoveries = 0` (`LGD = 1,0000`), haciendo que la **mediana de LGD en 2014 sea exactamente `1,0000`**.
4. **En 2015**, el **`88,97 %`** de los castigados no tiene ninguna recuperación registrada (`LGD media = 0,9807`), porque acaban de ser castigados semanas o pocos meses antes de diciembre de 2015.

Además, sobre los otros dos estados malos (`Default` con `1 219` casos y `Late (31-120 days)` con `11 591` casos), el **100,00 % tiene `recoveries = 0,00`** porque en Lending Club la cobranza post-castigo solo se contabiliza tras pasar al estado `Charged Off`.

---

## 4. Evaluación Crítica de Alternativas de Partición (Bajo Control del Equipo)

Dado que tenemos control sobre `parte_1/configs/pd.yaml` y `parte_2/configs/lgd.yaml`, existe una **tensión fundamental en este dataset** entre dos fenómenos opuestos:
- **Fenómeno A (Madurez del desenlace):** los años más antiguos (`2007–2012`) tienen desenlaces completos y cobranzas maduras (`~75 %` con recuperación).
- **Fenómeno B (Crecimiento exponencial de Lending Club):** los primeros años (`2007–2011`) tienen muy pocos préstamos (`40 534` cerrados en 5 años sumados, frente a `49 813` solo en `2012`, `73 194` en `2013` y `74 154` en `2014`).

Comparemos con números exactos de `data/raw/loan.csv` las **4 estrategias posibles**:

| Estrategia de Partición | Conjunto | Rango de Fechas (`issue_d`) | Créditos Cerrados (PD) | Malos PD (`% malos`) | Castigados (LGD) | `% Castigados con rec. > 0` |
|---|---|---|---:|---:|---:|---:|
| **Opción 1 (Actual)** | Entrenamiento<br>Validación<br>Prueba<br>*Excluido* | `2007-06` a `2012-12`<br>`2013-01` a `2013-12`<br>`2014-01` a `2014-12`<br>*`2015-01` a `2015-12`* | **90 347**<br>**73 194**<br>**74 154**<br>*30 835* | 14 729 (`16,30 %`)<br>16 798 (`22,95 %`)<br>19 441 (`26,22 %`)<br>*7 851 (`25,46 %`)* | **14 419**<br>**14 836**<br>**13 981**<br>*2 773* | **74,12 %**<br>**57,10 %**<br>**37,28 %**<br>*11,03 %* |
| **Opción 2 (Alt. A: Solo hasta 2013)** | Entrenamiento<br>Validación<br>Prueba<br>*Excluido* | `2007-06` a `2011-12`<br>`2012-01` a `2012-12`<br>`2013-01` a `2013-12`<br>*`2014-01` a `2015-12`* | **40 534**<br>**49 813**<br>**73 194**<br>*104 989* | 6 395 (`15,78 %`)<br>8 334 (`16,73 %`)<br>16 798 (`22,95 %`)<br>*27 292 (`26,00 %`)* | **6 335**<br>**8 084**<br>**14 836**<br>*16 754* | **73,20 %**<br>**74,84 %**<br>**57,10 %**<br>*32,93 %* |
| **Opción 3 (Alt. B: Corte en 2014-H1)** | Entrenamiento<br>Validación<br>Prueba<br>*Excluido* | `2007-06` a `2013-06`<br>`2013-07` a `2013-12`<br>`2014-01` a `2014-06`<br>*`2014-07` a `2015-12`* | **123 432**<br>**40 109**<br>**39 166**<br>*65 823* | 22 151 (`17,95 %`)<br>9 376 (`23,38 %`)<br>9 953 (`25,41 %`)<br>*17 339 (`26,34 %`)* | **21 177**<br>**8 078**<br>**7 719**<br>*9 035* | **70,27 %**<br>**52,96 %**<br>**42,92 %**<br>*23,36 %* |
| **Opción 4 (Alt. C: 2014 partido)** | Entrenamiento<br>Validación<br>Prueba<br>*Excluido* | `2007-06` a `2013-12`<br>`2014-01` a `2014-06`<br>`2014-07` a `2014-12`<br>*`2015-01` a `2015-12`* | **163 541**<br>**39 166**<br>**34 988**<br>*30 835* | 31 527 (`19,28 %`)<br>9 953 (`25,41 %`)<br>9 488 (`27,12 %`)<br>*7 851 (`25,46 %`)* | **29 255**<br>**7 719**<br>**6 262**<br>*2 773* | **65,49 %**<br>**42,92 %**<br>**30,33 %**<br>*11,03 %* |

---

## 5. Pros, Contras y Diagnóstico de Cada Opción

### Opción 1 — Partición Actual (`Train 2007–2012 | Val 2013 | Test 2014 | Excl 2015`)
- **Por qué se eligió originalmente:**
  1. Entrena exclusivamente con la ventana histórica donde los préstamos de 36 meses cumplieron su ciclo completo (`2007–2012`, `90 347` créditos cerrados y `74,12 %` de castigados con cobranza madura).
  2. Asigna un año calendario completo y distinto para validación (`2013`) y otro para prueba (`2014`), logrando un equilibrio casi `1:1:1` en castigados de LGD (`14 419` / `14 836` / `13 981`) y volúmenes grandes en PD (`90k / 73k / 74k`).
- **Debilidad real:**
  - En `Test (2014)`, el `68,53 %` de los créditos sigue en `Current` (censura de selección hacia prepagos y fallos tempranos, subiendo la tasa de malos a `26,22 %`) y en LGD el `62,72 %` de los castigados aún no registra recuperaciones (`mediana LGD = 1,00`). Eso explica numéricamente por qué en prueba la PD predicha (`0,196`) queda por debajo de la tasa observada (`0,262`) y por qué el LGD observado en 2014 (`0,945`) es más alto que el predicho (`0,923`).

### Opción 2 — Alternativa A (`Train 2007–2011 | Val 2012 | Test 2013 | Excl 2014–2015`)
- **Ventaja:** elimina por completo el año `2014`. Tanto validación (`2012`) como prueba (`2013`) tienen recuperaciones mucho más completas (`74,84 %` y `57,10 %`).
- **Problema grave:** como Lending Club era muy pequeño antes de 2012, el conjunto de **entrenamiento se queda con solo `40 534` créditos (`6 335` castigados para LGD)**, mientras que la prueba (`2013`) tiene **`73 194` créditos (`14 836` castigados)**. Entrenar una red neuronal con `6 335` datos para evaluarla en `14 836` invierte la proporción normal de aprendizaje.

### Opción 3 — Alternativa B (`Train 2007–2013H1 | Val 2013H2 | Test 2014H1 | Excl 2014H2–2015`)
- **Ventaja:** gana más datos de entrenamiento (`123 432` en PD; `21 177` en LGD) y elimina el segundo semestre de 2014 (`2014-H2`, que solo llevaba 12–17 meses de vida y tenía apenas `30,33 %` de recuperaciones).
- **Desventaja:** sigue teniendo censura importante en `2014-H1` (`25,41 %` malos y `42,92 %` con recuperación) y reduce a la mitad el tamaño de validación y prueba (`~39k` PD / `~8k` LGD).

### Opción 4 — Alternativa C (`Train 2007–2013 | Val 2014H1 | Test 2014H2 | Excl 2015`)
- **Descartada en `lgd.yaml`:** aunque maximiza el entrenamiento (`163 541` PD / `29 255` LGD), evalúa la prueba final en el segundo semestre de 2014 (`2014-H2`), donde el `69,67 %` de los castigados no tiene recuperación aún (`% con rec = 30,33 %`).

---

## 6. Conclusión y Verificación Empírica de Sensibilidad (`Opción 1` vs. `Opción 3`)

### 6.1 Comparación de desempeño predictivo (PD — Parte 1)
Para validar si ampliar el entrenamiento incorporando `2013-H1` (Opción 3: `123 432` créditos en entrenamiento, `40 109` en validación y `39 166` en prueba) producía mejoras sustanciales frente a la partición actual (Opción 1: `2007–2012 / 2013 / 2014`), se corrió una prueba controlada en [`auditoria_particion.py`](./auditoria_particion.py) cuyos resultados se guardaron en [`resultados/comparacion_opcion1_vs_opcion3.csv`](./resultados/comparacion_opcion1_vs_opcion3.csv):

| Esquema | Conjunto Evaluado | Modelo | $n$ | PD Media Pred. vs. Obs. | AUC | Gini | KS | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **Opción 1** (`Train 2007–2012`) | Validación `2013` | Reg. Logística | 73 194 | 0,1921 vs. 0,2295 | 0,6820 | 0,3641 | 0,2661 | 0,1662 |
| **Opción 1** (`Train 2007–2012`) | Validación `2013` | MLP `(64, 32)` | 73 194 | 0,1820 vs. 0,2295 | 0,6827 | 0,3655 | 0,2657 | 0,1665 |
| **Opción 1** (`Train 2007–2012`) | Prueba `2014` | Reg. Logística | 74 154 | 0,1967 vs. 0,2622 | 0,6818 | 0,3637 | 0,2648 | 0,1832 |
| **Opción 1** (`Train 2007–2012`) | Prueba `2014` | MLP `(64, 32)` | 74 154 | 0,1861 vs. 0,2622 | 0,6803 | 0,3606 | 0,2615 | 0,1847 |
| **Opción 1** (`Train 2007–2012`) | Prueba `2014-H1` | MLP `(64, 32)` | 39 166 | 0,1862 vs. 0,2541 | 0,6738 | 0,3476 | 0,2512 | 0,1810 |
| **Opción 3** (`Train 2007–2013H1`) | Validación `2013-H2` | MLP `(64, 32)` | 40 109 | 0,2103 vs. 0,2338 | 0,6861 | 0,3721 | 0,2719 | 0,1662 |
| **Opción 3** (`Train 2007–2013H1`) | Prueba `2014-H1` | MLP `(64, 32)` | 39 166 | 0,2150 vs. 0,2541 | 0,6825 | 0,3650 | 0,2651 | 0,1764 |

### 6.2 Decisión final del equipo
1. **Se mantiene oficialmente la Opción 1 (`2007–2012 / 2013 / 2014`):** la ganancia de discriminación al pasar a la Opción 3 es marginal (`AUC` de `0,6803` en `2014` completo frente a `0,6825` en `2014-H1`), mientras que la Opción 1 preserva la maduración completa de todos los créditos de 36 meses en entrenamiento, evalúa años calendario completos sin estacionalidad y mantiene 100 % sincronizados los modelos ya entrenados de la Parte 2 (LGD).
2. **Aporte para el reporte y la sustentación:** la figura [`fig0_censura_y_tasa_malos_por_anio.png`](./figuras/fig0_censura_y_tasa_malos_por_anio.png) y esta tabla documentan por qué se excluye 2015, por qué la censura parcial de 2014 explica la brecha de calibración (`0,1861` vs. `0,2622`) y qué ocurriría si se desplazara la fecha de corte en una modificación en vivo (`--set`).
