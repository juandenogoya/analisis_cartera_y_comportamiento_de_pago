# Status del Proyecto: Analisis de Cartera y Comportamiento de Pago

## Resumen del Estado Actual

El proyecto se encuentra en una fase avanzada de desarrollo, con un flujo de trabajo completo que abarca desde la carga de datos hasta la optimización de hiperparámetros de un modelo de Random Forest.

### Logros Identificados
1.  **Procesamiento de Datos:**
    *   Se ha implementado una limpieza de datos robusta, filtrando registros inválidos (ingresos '0', 'NC') y estados no deseados ('CANCELADO', etc.).
    *   Se han tratado valores nulos en columnas clave.
    *   El dataset final cuenta con aproximadamente 214,000 registros.
2.  **Análisis Exploratorio (EDA):**
    *   Se han generado visualizaciones útiles sobre la distribución de clientes por canal, sexo y comportamiento de pago.
    *   Se han calculado tasas de aprobación por canal.
3.  **Preprocesamiento:**
    *   Se ha aplicado **One-Hot Encoding** a variables categóricas.
    *   Se ha aplicado **StandardScaler** a variables numéricas.
    *   Se ha realizado un split estratificado de Train/Test.
4.  **Modelado:**
    *   Se ha implementado un modelo **Random Forest Classifier**.
    *   Se han aplicado técnicas de búsqueda de hiperparámetros (**Random Search** y **Bayesian Search**) logrando una alta accuracy global (~94-96%).

### Hallazgos Críticos y Áreas de Riesgo
1.  **Data Leakage (Fuga de Información):**
    *   **Problema:** El modelo actual utiliza variables como `CD_CAJON_MORA` (cajón de mora actual), `NU_CUOTAS_EN_MORA` (número de cuotas en mora) y `NU_CUOTA_PAGADAS` para predecir `CD_SUBESTADO_PRODUCTO`.
    *   **Impacto:** Estas variables definen el estado actual del cliente. Usarlas para predecir ese mismo estado no constituye un modelo *predictivo* de riesgo futuro, sino un modelo *descriptivo* del estado presente. Esto infla artificialmente las métricas de rendimiento (accuracy) y hace que el modelo no sea útil para predecir el riesgo *antes* de que ocurra la mora.
2.  **Desbalanceo de Clases:**
    *   **Problema:** La clase mayoritaria ("AL DIA") domina el dataset.
    *   **Impacto:** El reporte de clasificación muestra una precisión y recall de **0.00** para clases minoritarias críticas como "PROBLEMAS", "VENDIDO" y "WRITE OFF". El modelo está sesgado hacia la clase mayoritaria.
3.  **Métricas Engañosas:**
    *   Debido al desbalance, la métrica de *Accuracy* no es fiable. Un modelo que prediga siempre "AL DIA" tendría una accuracy alta pero sería inútil para detectar riesgo.

---

## Plan de Mejora

El siguiente plan establece las etapas para transformar el proyecto actual en una solución predictiva robusta y útil para el negocio.

### Etapa 1: Integridad de Datos y Baseline Real (Prioridad Alta)
**Objetivo:** Eliminar el data leakage para entender la capacidad real del modelo para predecir riesgo futuro.
1.  **Identificar y Eliminar Variables de Fuga:**
    *   Remover variables que contienen información del futuro o del estado actual de mora: `CD_CAJON_MORA`, `NU_CUOTAS_EN_MORA`, `NU_DIAS_MORA` (si se usa).
2.  **Re-entrenar y Evaluar:**
    *   Entrenar el Random Forest con el set de datos "limpio" de fugas.
    *   Establecer un nuevo baseline de métricas. Es esperado que la accuracy baje significativamente, pero este será el rendimiento real.

### Etapa 2: Gestión del Desbalanceo (Prioridad Alta)
**Objetivo:** Mejorar la capacidad del modelo para detectar las clases minoritarias (clientes con problemas).
1.  **Técnicas de Resampling:**
    *   Implementar **SMOTE** (Synthetic Minority Over-sampling Technique) para el set de entrenamiento.
    *   Evaluar **Undersampling** de la clase mayoritaria.
2.  **Ajuste de Pesos:**
    *   Utilizar el parámetro `class_weight='balanced'` en el Random Forest.
3.  **Nuevas Métricas:**
    *   Dejar de usar Accuracy como métrica principal.
    *   Optimizar para **F1-Score (Macro/Weighted)** o **ROC-AUC**.

### Etapa 3: Ingeniería y Selección de Features (Prioridad Media)
**Objetivo:** Enriquecer la información disponible para mejorar la predicción.
1.  **Análisis de Importancia:**
    *   Utilizar `feature_importances_` para identificar las variables que realmente aportan valor predictivo (sin las variables de fuga).
2.  **Creación de Features:**
    *   Si es posible, derivar nuevas variables (e.g., ratios deuda/ingreso si la información lo permite, categorización de edad).

### Etapa 4: Comparación de Modelos (Prioridad Media)
**Objetivo:** Validar si Random Forest es el mejor algoritmo para este problema.
1.  **Benchmark:**
    *   Probar algoritmos de Boosting como **XGBoost**, **LightGBM** o **CatBoost**, que suelen tener mejor rendimiento en datos tabulares desbalanceados.
    *   Probar un modelo lineal simple (**Logistic Regression**) como línea base.

### Etapa 5: Refactorización y Modularización (Prioridad Baja)
**Objetivo:** Mejorar la calidad del código y su mantenibilidad.
1.  **Modularización:**
    *   Mover las funciones de carga, limpieza y entrenamiento a archivos `.py` separados.
    *   Dejar el Notebook solo para orquestación y visualización de resultados.
