# 🏦 Marketing bancario con Machine Learning

Modelo de clasificación que predice si un cliente **contratará un depósito a plazo** tras una campaña de marketing telefónico de un banco portugués. El preprocesado y el modelo se encadenan en un único **Pipeline de scikit-learn**.

📓 Notebook: [`bank_marketing_pipe.ipynb`](bank_marketing_pipe.ipynb)

---

## 🎯 El problema

Las campañas de telemarketing son caras: cada llamada cuesta tiempo y dinero, y la gran mayoría de clientes no contrata. Si el banco supiera de antemano qué clientes tienen más probabilidad de decir que sí, podría **concentrar las llamadas en ellos**.

La dificultad está en que el dataset está **muy desbalanceado**: solo el **11,7 %** de los clientes contrató el depósito. Un modelo que dijera siempre "no" acertaría el 88 % de las veces, pero no serviría para nada. Por eso, más que la *accuracy*, la métrica clave es el **recall de la clase "sí"**: qué porcentaje de los clientes interesados consigue detectar el modelo.

---

## 📊 Dataset

| | |
|---|---|
| **Fuente** | [Bank Marketing – UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) (fichero `bank-full.csv`) |
| **Registros** | 45.211 clientes |
| **Variables** | 16 *features* + 1 objetivo (`y`: `yes` / `no`) |
| **Valores nulos** | Ninguno |
| **Licencia** | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Las variables incluyen datos del cliente (edad, profesión, estado civil, educación, saldo, préstamos), del último contacto de la campaña (tipo de contacto, día, mes, duración de la llamada) y de campañas anteriores (número de contactos, resultado previo).

> **Referencia:** S. Moro, P. Cortez y P. Rita (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems, 62, 22-31.

---

## ⚙️ Enfoque

1. **Análisis exploratorio**: dimensiones, tipos de datos, estadísticos descriptivos y comprobación de valores nulos.
2. **Separación de variables**: se excluye `duration` (duración de la llamada), porque solo se conoce cuando la llamada ya ha terminado y usarla sería *data leakage*. Quedan 6 numéricas y 9 categóricas, detectadas automáticamente por tipo de dato.
3. **División train / test**: 80 % / 20 %, con semilla fija (`random_state=22`) para que los resultados sean reproducibles.
4. **Preprocesado con `ColumnTransformer`**:
   - Numéricas → `MinMaxScaler`
   - Categóricas → `SimpleImputer` + `OneHotEncoder(handle_unknown="ignore")`
5. **Modelos**, ambos con `class_weight="balanced"` para compensar el desbalanceo:
   - `DecisionTreeClassifier`
   - `RandomForestClassifier` (150 árboles, `min_samples_split=0.3`)
6. **Evaluación** con `classification_report`, matrices de confusión, curva de ganancia e importancia de las variables.

Al integrar el preprocesado y el modelo en un único `Pipeline`, las transformaciones se ajustan **solo con los datos de entrenamiento**. Así se evita la fuga de información (*data leakage*) hacia el conjunto de test, y el modelo completo se puede reutilizar con datos nuevos en una sola llamada a `predict()`.

```
Pipeline
├── ColumnTransformer
│   ├── num → MinMaxScaler
│   └── cat → SimpleImputer → OneHotEncoder
└── Classifier (Decision Tree / Random Forest)
```

---

## 📈 Resultados

Sobre el conjunto de test (9.043 clientes, de los cuales 1.072 contrataron):

| Modelo | Accuracy | Precision (sí) | **Recall (sí)** | F1 (sí) | Clientes detectados | Clientes perdidos |
|---|---|---|---|---|---|---|
| Decision Tree | 0,83 | 0,30 | 0,31 | 0,30 | 329 | 743 |
| **Random Forest** | 0,60 | 0,19 | **0,72** | 0,30 | **773** | **299** |

**Conclusión:** aunque el Decision Tree tiene mejor *accuracy* y ambos empatan en F1, el **Random Forest es el modelo más útil para la campaña**. Detecta al **72 % de los clientes interesados** (773 de 1.072), frente al 31 % del Decision Tree. A cambio, hace muchas llamadas a clientes que finalmente no contratarán: su precisión es del 19 %, frente al 11,9 % de contrataciones en el test si se llama al azar. Usando su probabilidad para priorizar, **llamar solo al 30 % de los clientes con mayor probabilidad capta el 62 % de las contrataciones**, aproximadamente el doble que llamando al azar.

> Los resultados pueden variar ligeramente (±1 punto) según la versión de scikit-learn instalada.

---

## 🧪 Limitaciones y próximos pasos

- **La precisión sigue siendo baja** (19 %): el modelo sirve para ordenar a quién llamar primero, no para asegurar ventas. La importancia de las variables indica en qué se apoya el modelo, no qué causa la contratación.
- Usar un **split estratificado** (`stratify=y`) para mantener la misma proporción de "sí" en train y test.
- **Ajustar hiperparámetros** con `GridSearchCV` o `RandomizedSearchCV` y validación cruzada.
- Probar otros modelos (Logistic Regression, Gradient Boosting) y ajustar el **umbral de decisión** para equilibrar recall y precisión según el coste real de cada llamada.

---

## 🚀 Cómo ejecutarlo

### 1. Clonar el repositorio

```bash
git clone https://github.com/juan1639/Bank_marketing_notebook_portfolio.git
cd Bank_marketing_notebook_portfolio
```

### 2. Crear un entorno virtual e instalar dependencias

En Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En Linux o macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Abrir el notebook

```bash
jupyter notebook bank_marketing_pipe.ipynb
```

El dataset `bank-full.csv` ya está incluido en el repositorio, junto al notebook. También se puede abrir directamente en **Google Colab** subiendo el notebook y el CSV.

---

## 📁 Estructura del proyecto

```
Bank_marketing_notebook_portfolio/
├── bank_marketing_pipe.ipynb   # Notebook con el análisis y los modelos
├── bank-full.csv               # Dataset (UCI, CC BY 4.0)
├── requirements.txt            # Dependencias
├── LICENSE                     # Licencia del código (MIT)
└── README.md
```

---

## 🛠️ Tecnologías

Python · pandas · scikit-learn · matplotlib · Jupyter

---

## 👤 Autor

**Juan Eguía Abad** · [GitHub](https://github.com/juan1639) · [Portfolio](https://juan1639.github.io/Portfolio_data_science/)

El código de este repositorio se distribuye bajo licencia MIT. El dataset pertenece a sus autores y se distribuye bajo licencia CC BY 4.0.
