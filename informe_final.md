# Clasificación Automática de Sentimiento en Tweets sobre Marcas y Productos

**Autor:** Fernando Ureta  
**Ramo:** Mineria de datos 
**Profesor:** Eliecer Peña
**Fecha de entrega:** 21 de junio de 2026  

---

## Resumen

Este proyecto aborda la clasificación automática de sentimiento en tweets sobre marcas y productos tecnológicos. El problema se formula como una tarea de clasificación multiclase supervisada con cuatro categorías: Positive, Negative, Neutral e Irrelevant. Se utilizó el dataset *Twitter Entity Sentiment Analysis*, compuesto por 72.139 tweets tras el proceso de limpieza. La metodología combina representaciones textuales mediante TF-IDF con 14 features tabulares estructurales del tweet, integrando señales semánticas y lingüísticas en un único pipeline sin filtración de información. Se entrenaron y compararon tres modelos: Multinomial Naive Bayes, Regresión Logística y LightGBM. El modelo seleccionado, LightGBM, alcanzó un F1-macro de 0.7515 sobre el conjunto de test completamente reservado, con una diferencia de apenas 0.0004 respecto al conjunto de validación, lo que evidencia una generalización robusta. La principal limitación identificada es la dependencia del predictor Entity (marca del tweet) en el conjunto de entrenamiento, que no siempre está disponible en producción. La solución se encuentra disponible como demo interactiva en: https://twitter-sentiment-entrega3-7qmdjsm5h4vuytvqujqick.streamlit.app/

---

## Contenido

1. Introducción
   - 1.1 Contexto y planteamiento del problema
   - 1.2 Pregunta del proyecto
   - 1.3 Objetivos
   - 1.4 Alcance y límites iniciales
2. Antecedentes y marco conceptual
3. Metodología
   - 3.1 Dataset, fuente y unidad de análisis
   - 3.2 Preparación y calidad de los datos
   - 3.3 Separación de datos y prevención de leakage
   - 3.4 Modelo desarrollado
4. Resultados
5. Discusión
6. Conclusiones
7. Referencias

---

## 1. Introducción

### 1.1 Contexto y planteamiento del problema

Las redes sociales, y Twitter en particular, se han convertido en una fuente masiva de opinión pública sobre productos, marcas y servicios. Las empresas invierten recursos considerables en monitorear la percepción que tienen sus consumidores, pero el volumen de tweets generados diariamente hace inviable la revisión manual. La clasificación automática de sentimiento permite procesar miles de tweets en segundos, identificando si el tono de una publicación es positivo, negativo, neutral hacia la marca, o simplemente irrelevante para el análisis.

El desafío técnico de este problema radica en que el sentimiento en texto informal (abreviaciones, emojis, sarcasmo, menciones directas) es difícil de capturar con reglas heurísticas. Los modelos de aprendizaje automático entrenados sobre datos etiquetados ofrecen una alternativa escalable y adaptable.

### 1.2 Pregunta del proyecto

¿Es posible clasificar automáticamente el sentimiento de tweets sobre marcas y productos en cuatro categorías (Positive, Negative, Neutral, Irrelevant) con un F1-macro superior a 0.70, utilizando una combinación de representaciones textuales y features estructurales del tweet?

### 1.3 Objetivos

**Objetivo general:** Desarrollar un modelo de clasificación multiclase de sentimiento en tweets sobre marcas que alcance un F1-macro ≥ 0.70 en datos no vistos durante el entrenamiento.

**Objetivos específicos:**

1. Construir un pipeline de preprocesamiento reproducible que integre features textuales (TF-IDF) y features tabulares estructurales del tweet, garantizando la ausencia de filtración de información entre conjuntos.
2. Entrenar y comparar al menos tres modelos de clasificación con distintas capacidades expresivas (Naive Bayes, Regresión Logística y LightGBM), evaluados con F1-macro como métrica principal.
3. Evaluar el modelo seleccionado sobre un conjunto de test reservado de forma estricta, reportando métricas completas por clase y analizando los patrones de error.
4. Demostrar el funcionamiento del modelo en producción mediante una aplicación interactiva de acceso público que procese tweets nuevos en tiempo real.
5. Identificar las limitaciones del modelo y los escenarios en que podría fallar, con especial atención a las clases minoritarias y a la ausencia de información de entidad en inferencia real.

### 1.4 Alcance y límites iniciales

El proyecto abarca el entrenamiento, evaluación y despliegue de un modelo de clasificación de sentimiento sobre texto en inglés, limitado a las cuatro categorías del dataset original. El modelo fue diseñado para el dominio tecnológico/empresarial (marcas como Microsoft, Verizon, Apple, etc.) y su rendimiento puede degradarse en dominios distintos. No se abordan tareas relacionadas como detección de sarcasmo, análisis de aspecto o clasificación de idiomas distintos al inglés. Los resultados deben interpretarse en el contexto de tweets cortos y de naturaleza informal; el modelo no fue validado en textos largos ni en otros formatos.

---

## 2. Antecedentes y marco conceptual

**Análisis de sentimiento** es la tarea de identificar y extraer opiniones subjetivas de textos. En su versión multiclase, se distingue entre polaridades positiva, negativa, neutral, y categorías adicionales como "irrelevante", donde el texto no hace referencia al tema de interés.

**TF-IDF** (*Term Frequency – Inverse Document Frequency*) es una representación vectorial del texto que asigna mayor peso a términos frecuentes en un documento pero poco frecuentes en el corpus completo. Captura la relevancia de cada término para distinguir documentos entre sí. En este proyecto se utilizaron unigramas y bigramas (combinaciones de una y dos palabras consecutivas) con un vocabulario de 10.000 términos.

**Reducción de dimensionalidad con SVD truncado** (*Truncated Singular Value Decomposition*) permite comprimir la representación TF-IDF de 10.000 dimensiones a un espacio denso de menor tamaño (200 componentes), preservando las direcciones de mayor varianza. Esta transformación es necesaria para que algoritmos basados en árboles como LightGBM puedan operar eficientemente.

**Gradient Boosting** es una técnica de ensamble que construye secuencialmente árboles de decisión, donde cada árbol corrige los errores del anterior. LightGBM es una implementación optimizada para CPU, con crecimiento de árboles hoja por hoja (*leaf-wise*) que resulta más eficiente que el crecimiento nivel por nivel tradicional. Su capacidad para modelar interacciones no lineales entre features lo hace especialmente adecuado cuando las señales discriminativas son combinaciones de variables.

**F1-macro** es la media no ponderada del F1-score de cada clase. A diferencia de la accuracy, no se ve inflada por clases mayoritarias, lo que la hace apropiada cuando el objetivo es que el modelo funcione bien en todas las clases por igual, incluidas las minoritarias.

---

## 3. Metodología

### 3.1 Dataset, fuente y unidad de análisis

El dataset utilizado es *Twitter Entity Sentiment Analysis*, disponible públicamente en Kaggle. Está compuesto por dos archivos CSV: uno de entrenamiento y uno de validación, que en conjunto suman 75.682 tweets originales.

Cada observación (unidad de análisis) es un tweet individual, caracterizado por:
- **ID:** identificador único
- **Entity:** marca o entidad mencionada (32 marcas distintas, ej. Microsoft, Verizon, Apple)
- **Sentiment:** etiqueta de sentimiento (variable objetivo)
- **Tweet:** texto completo del tweet en inglés

La distribución de clases en el dataset limpio es:

| Clase | Frecuencia | Proporción |
|---|---|---|
| Negative | 21.773 | 30.2% |
| Positive | 19.829 | 27.5% |
| Neutral | 17.879 | 24.8% |
| Irrelevant | 12.658 | 17.5% |

El dataset presenta un desbalance leve entre clases, con Irrelevant como la clase minoritaria (17.5%). Este desbalance motiva el uso de F1-macro como métrica principal en lugar de accuracy.

### 3.2 Preparación y calidad de los datos

**Limpieza inicial:** Se eliminaron 686 filas con el campo Tweet nulo y 2.857 filas duplicadas exactas, resultando en 72.139 tweets para el modelado.

**Extracción de features tabulares:** Se extrajeron 11 features numéricas directamente del texto del tweet:

| Feature | Descripción |
|---|---|
| tweet_length | Longitud total en caracteres |
| word_count | Número de palabras |
| avg_word_length | Longitud promedio de palabras |
| n_exclamations | Número de signos de exclamación |
| n_questions | Número de signos de interrogación |
| n_uppercase | Cantidad de letras mayúsculas |
| uppercase_ratio | Proporción de mayúsculas sobre letras totales |
| n_hashtags | Número de hashtags (#) |
| n_mentions | Número de menciones (@) |
| n_urls | Número de URLs |
| has_negation | Presencia de palabras de negación (not, never, no, etc.) |

**Codificación de Entity:** La variable Entity (32 marcas) fue codificada con MEstimate Encoder en esquema one-vs-rest, generando 3 columnas numéricas que representan la probabilidad de cada sentimiento (excepto Positive como categoría de referencia). Este encoder evita el sobreajuste que produce One-Hot Encoding con variables de alta cardinalidad.

**Tratamiento de outliers:** Se aplicó el método IQR sobre 5 columnas continuas, reemplazando valores extremos por la mediana calculada exclusivamente en el conjunto de entrenamiento.

**Imputación:** Se aplicó IterativeImputer con CatBoost de forma defensiva; no existían nulos reales en los datos procesados.

**Vectorización TF-IDF:** El texto fue vectorizado con TfidfVectorizer configurado con 10.000 features máximas, bigramas, frecuencia mínima de documento de 3, frecuencia máxima de 95%, y transformación sublineal de TF para reducir el impacto de términos muy frecuentes.

**Escalado:** Las features tabulares fueron escaladas con MaxAbsScaler, que divide por el valor absoluto máximo de cada columna. Esta elección preserva la estructura sparse de las matrices al no centrar los datos.

### 3.3 Separación de datos y prevención de leakage

Los datos fueron divididos en tres conjuntos mediante muestreo aleatorio estratificado (por clase) con semilla fija `random_state=42`:

| Conjunto | Filas | Proporción |
|---|---|---|
| Train | 50.497 | 70% |
| Validación | 10.821 | 15% |
| Test | 10.821 | 15% |

La estratificación garantiza que la distribución de clases se mantiene en los tres conjuntos, preservando el desbalance original.

**Prevención de leakage:** Todas las transformaciones que aprenden estadísticas de los datos (TF-IDF, MaxAbsScaler, TruncatedSVD, MEstimate Encoder, IQR de outliers, mediana de imputación) fueron ajustadas (*fit*) exclusivamente sobre el conjunto de entrenamiento y luego aplicadas (*transform*) sin re-ajuste sobre validación y test. El conjunto de test fue utilizado una única vez al final del proyecto, únicamente para reportar la performance del modelo seleccionado.

No se utilizaron variables proxy del target ni información futura en ninguna de las features. La columna Entity no representa el target; es información disponible en el momento de la observación.

### 3.4 Modelo desarrollado

Se entrenaron tres modelos con distintas capacidades expresivas para identificar la combinación de features y algoritmo que maximiza el F1-macro en validación.

**Modelo 1 — Multinomial Naive Bayes:** Clasificador probabilístico generativo que opera directamente sobre la matriz TF-IDF sparse. Sirve como baseline de referencia para NLP clásico.

**Modelo 2 — Regresión Logística Multiclase:** Clasificador discriminativo lineal entrenado sobre la combinación de TF-IDF y las 14 features tabulares en formato sparse. Permite cuantificar el aporte de las features estructurales sobre el texto puro.

**Modelo 3 — LightGBM:** Modelo de gradient boosting entrenado sobre la combinación densa de SVD(TF-IDF, 200 componentes) y las 14 features tabulares. Captura interacciones no lineales entre features. Se utilizó `class_weight='balanced'` para compensar el desbalance de clases y early stopping con paciencia de 30 iteraciones sobre el conjunto de validación para evitar sobreajuste.

El modelo seleccionado para producción es **LightGBM**, por obtener el mayor F1-macro en validación y la mayor estabilidad val→test. Se encuentra disponible como aplicación interactiva en:

**Demo en vivo:** https://twitter-sentiment-entrega3-7qmdjsm5h4vuytvqujqick.streamlit.app/

La aplicación permite ingresar cualquier tweet en inglés y obtiene en tiempo real la clase predicha junto con el gráfico de probabilidades para las cuatro clases.

**Código fuente:** https://github.com/fernandoureta/twitter-sentiment-entrega3

---

## 4. Resultados

### Comparación de modelos en validación

La Tabla 2 resume el rendimiento de los tres modelos evaluados sobre el conjunto de validación.

**Tabla 2.** Comparación de métricas en conjunto de validación.

| Modelo | F1-macro | F1-weighted | Accuracy |
|---|---|---|---|
| Multinomial Naive Bayes | 0.6268 | 0.6412 | 0.6521 |
| Logistic Regression | 0.7199 | 0.7286 | 0.7303 |
| **LightGBM** | **0.7519** | **0.7559** | **0.7555** |

Los tres modelos superan ampliamente el baseline aleatorio estratificado (~0.25 F1-macro). La progresión NB → LR → LightGBM evidencia que cada componente del pipeline aporta señal incremental.

**Multinomial Naive Bayes** obtuvo F1-macro de 0.6268. El principal problema es la clase Irrelevant, con recall de 0.38: el modelo solo detecta 4 de cada 10 tweets irrelevantes. Al operar únicamente sobre texto, NB no puede aprovechar la señal de la entidad, que es el predictor más fuerte según el análisis exploratorio.

**Logistic Regression** mejoró en +0.093 F1-macro respecto a NB, pasando de 0.627 a 0.720. Esta ganancia confirma que las features tabulares —particularmente los Entity encodings— aportan señal discriminativa que el texto solo no captura. El recall de Irrelevant subió de 0.38 a 0.61, la mayor mejora individual de esta transición.

**LightGBM** alcanzó F1-macro de 0.7519, superando a LR en +0.032. La mejora más notable se concentra en Irrelevant, cuyo recall aumentó de 0.61 a 0.78, lo que indica que las interacciones no lineales entre features son especialmente útiles para esta clase sin vocabulario propio marcado.

### Resultados por clase del modelo final (validación)

**Tabla 3.** Métricas por clase de LightGBM en validación.

| Clase | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Irrelevant | 0.67 | 0.78 | 0.72 | 1.893 |
| Negative | 0.78 | 0.78 | 0.78 | 3.269 |
| Neutral | 0.76 | 0.73 | 0.74 | 2.682 |
| Positive | 0.78 | 0.74 | 0.76 | 2.977 |
| **Macro avg** | **0.75** | **0.76** | **0.75** | **10.821** |

La brecha entre F1-macro (0.7519) y F1-weighted (0.7559) es mínima (Δ = 0.004), lo que indica que el modelo trata las cuatro clases de forma equilibrada a pesar del desbalance. El parámetro `class_weight='balanced'` evitó que la clase Irrelevant (17.5% del dataset) quedara relegada.

### Evaluación final en test

El modelo LightGBM fue evaluado una única vez sobre el conjunto de test, obteniendo los resultados de la Tabla 4.

**Tabla 4.** Métricas por clase de LightGBM en test.

| Clase | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Irrelevant | 0.66 | 0.76 | 0.71 | 1.894 |
| Negative | 0.78 | 0.79 | 0.78 | 3.268 |
| Neutral | 0.78 | 0.74 | 0.76 | 2.682 |
| Positive | 0.77 | 0.74 | 0.75 | 2.977 |
| **Macro avg** | **0.75** | **0.76** | **0.75** | **10.821** |

El F1-macro en test fue de **0.7515**, prácticamente idéntico al 0.7519 en validación (diferencia de −0.0004). Esta estabilidad excepcional confirma que no hubo sobreajuste al proceso de selección de modelos y que el pipeline generaliza correctamente a datos completamente no vistos. El criterio de éxito definido (F1-macro ≥ 0.70) fue alcanzado con margen.

### Verificación del modelo con tweets nuevos

Para confirmar el funcionamiento end-to-end del pipeline, se evaluaron 8 tweets inventados con sentimiento esperado claro. El modelo clasificó correctamente 5 de 8, con errores concentrados en la clase Neutral e Irrelevant —clases sin vocabulario propio marcado—, comportamiento coherente con los resultados cuantitativos del test.

---

## 5. Discusión

**Interpretación de los resultados:** El modelo alcanzó F1-macro de 0.75 en datos no vistos, superando el umbral objetivo de 0.70. La clase Negative es la mejor detectada en todos los modelos (recall consistente ~0.78–0.82), lo cual tiene sentido dado que palabras como "worst", "terrible" o "never again" son señales lexicales fuertes y frecuentes. La clase Irrelevant es la más difícil porque sus tweets no tienen un vocabulario propio: pueden tratar cualquier tema no relacionado con la marca, y el modelo debe inferir irrelevancia principalmente a partir de la entidad y los marcadores estructurales, no del texto.

**Utilidad y casos de uso:** La solución es útil para equipos de marketing y gestión de reputación que necesitan monitorear grandes volúmenes de menciones de marca en Twitter. Permite priorizar la atención humana hacia tweets negativos y clasificar automáticamente el volumen diario de menciones sin revisión manual.

**Limitaciones del dataset:** El dataset proviene de una fuente pública con etiquetado humano, lo que implica subjetividad en los límites entre clases (especialmente Neutral vs. Irrelevant). La cobertura está limitada a 32 marcas del dominio tecnológico/entretenimiento, lo que puede reducir la generalización a otros dominios o marcas no representadas.

**Limitaciones del modelo en producción:** La feature más importante del modelo es la entidad (marca) del tweet. En la demo de producción, la entidad no se especifica y las features Entity_me_* se imputan como neutras, lo que explica que el modelo clasifique tweets neutrales o irrelevantes como Positive (Positive es la categoría de referencia del encoder). En un sistema de producción real, la entidad estaría disponible si se monitorea una marca específica.

**Riesgos de sesgo:** El modelo puede reproducir sesgos presentes en los datos de entrenamiento. Si ciertas marcas están sistemáticamente sobrerepresentadas en una clase, el modelo aprende esa asociación independientemente del contenido del tweet. Asimismo, el modelo fue entrenado en inglés; su aplicación a otros idiomas produciría resultados no confiables.

**Resultados confiables, no sospechosamente altos:** Un F1-macro de 0.75 es un resultado razonable para clasificación de sentimiento multiclase en texto informal. No se acerca a resultados sospechosos (>0.95); la diferencia de 0.0004 entre validación y test confirma que no hubo filtración de información.

---

## 6. Conclusiones

El proyecto respondió afirmativamente a la pregunta planteada: es posible clasificar automáticamente el sentimiento de tweets sobre marcas con F1-macro superior a 0.70 combinando TF-IDF con features tabulares estructurales. El modelo LightGBM alcanzó F1-macro de 0.7515 en test, superando el umbral objetivo con margen.

Los hallazgos principales son tres: primero, la señal de la entidad (marca) es el predictor más potente del sentimiento; sin ella, el F1-macro cae de 0.72 (LR) a 0.63 (NB). Segundo, las interacciones no lineales que captura LightGBM son especialmente valiosas para la clase Irrelevant, que pasa de recall 0.38 (NB) a 0.78 (LightGBM). Tercero, el pipeline construido es reproducible y estable: la diferencia de F1-macro entre validación y test es de apenas 0.0004, lo que indica ausencia de sobreajuste.

El principal límite del sistema es la dependencia de la información de entidad en el momento de inferencia. En un despliegue real, este problema se resuelve si el sistema de monitoreo conoce la marca que está siendo analizada. El modelo está disponible en producción y puede ser utilizado directamente desde el navegador sin conocimientos técnicos.

---

## Referencias

- Go, A., Bhayani, R., & Huang, L. (2009). *Twitter sentiment classification using distant supervision*. CS224N Project Report, Stanford University.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*.
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Dataset: Twitter Entity Sentiment Analysis. Kaggle. https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis
- Streamlit. (2024). *Streamlit — A faster way to build and share data apps*. https://streamlit.io
