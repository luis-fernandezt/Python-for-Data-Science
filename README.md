
# Python for Data Science

Resumen de apuntes de Python para ciencia de datos utilizando Jupyter Notebook y Visual Studio Code.

---

## 📦 Librerías utilizadas

### Instalación con pip:

```bash
pip install pandas matplotlib seaborn numpy statsmodels scikit-learn dataclasses getweather
```

### Librerías importadas:

- **pandas** → `import pandas as pd`
- **numpy** → `import numpy as np`
- **matplotlib** → `import matplotlib.pyplot as plt` o `import matplotlib.pyplot as pp`
- **seaborn** → `import seaborn as sns`
- **statsmodels** → `import statsmodels.api as sm` y `from statsmodels.formula.api import ols`
- **scikit-learn** → `from sklearn.model_selection import train_test_split`, `from sklearn.neighbors import KNeighborsClassifier`
- **datetime** → `from datetime import date, time, datetime, timedelta`
- **math** → `import math`
- **os / path** → `import os`, `from os import path`
- **shutil / zipfile** → `import shutil`, `from shutil import make_archive`, `from zipfile import ZipFile`
- **calendar** → `import calendar`
- **json / urllib** → `import json`, `import urllib.request`
- **collections** → `import collections`, `@dataclass`, `namedtuple`
- **itertools** → `import itertools`
- **getweather** → `import getweather` *(librería personalizada o externa no estándar)*

---

## 🛠️ Comandos y Sintaxis Útil

> Ordenados de menor a mayor complejidad y agrupados por funcionalidad.

### 📘 Entrada / Salida y estructuras básicas

```python
print("Hello World")  # salida simple
mylist = [0, 1, "two", 3.2, False]
mydict = {"one": 1, "two": 2}
del mylist
del mydict
```

### 📂 Archivos

```python
file = open("file.txt", "w+")  # escribir
file = open("file.txt", "a+")  # agregar
file = open("file.txt", "r")   # leer
file.readlines()
file.close()
```

### 📐 Tipos de datos y operaciones comunes

```python
len(data)
type(data)
sorted(data)
data.sort()
data.append("x")
data.remove("x")
data.insert(0, "x")
data.extend(["a", "b"])
```

### 🔍 Pandas: limpieza y exploración de datos

```python
pd.read_csv("data.csv")
data.head()
data.tail()
data.info()
data.shape
data.columns
data.dtypes
data.index
data.isna().sum()
data.dropna()
data.duplicated().sum()
data["col"].value_counts(normalize=True)
data["col"].dtype
```

### 🔎 Pandas: filtrado y agrupación

```python
data["Year"] == 2025
data.loc[data["Year"] == 2025, :]
data.groupby("Year").count()
data.sort_values("Count", ascending=False)
data.query("col > 10")
```

### 🧪 Estadística básica

```python
np.mean(data["value"])
np.min(data)
np.max(data)
np.nanmin(data)
np.nanmax(data)
np.isnan(data)
np.percentile(data, 2.5)
np.percentile(data, 97.5)
```

### 📊 Visualización

```python
data.plot.barh(x="Year", y="Count")
sns.barplot(x="col1", y="col2", data=data, ci=False)
sns.countplot(x="col", data=data)
sns.histplot(data=data, x="col", hue="group")
sns.pairplot(data)
plt.hist(data["col"])
plt.scatter(x=data["x"], y=data["y"], s=5)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend(["name1", "name2"])
plt.xlim(0, 100)
plt.axhline(0)
plt.show()
```

### 📈 Modelado

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn.predict(X_test)
knn.score(X_test, y_test)

model = ols(formula="value1 ~ value2", data=data).fit()
model.summary()
sm.qqplot(model.resid, line='s')
```

### 🧭 Fechas y tiempos

```python
datetime.now()
date.today()
datetime.now().strftime("%Y-%m-%d")
datetime.now() + timedelta(days=10)
datetime.now() - timedelta(weeks=1)
```

### 🌀 Control de flujo

```python
for x in range(10):
    print(x)

while x < 5:
    print(x)
    x += 1

days = ["Mon", "Tue", "Wed"]
for i, d in enumerate(days):
    print(i, d)

if x > 0:
    print("Positive")
elif x == 0:
    print("Zero")
else:
    print("Negative")
```

---

## 🌐 Webpages y recursos en línea

- [Documentación datetime Python](https://docs.python.org/3.6/library/datetime.html#strftime-and-strptime-behavior)
- [Documentación JSON Python](https://docs.python.org/es/3/library/json.html)
- [Kaggle](https://www.kaggle.com/)
- [USGS Earthquake GeoJSON feed](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php)
- [HTML Viewer](https://html.onlineviewer.net/)

---

## 📚 Referencias Bibliográficas

- McKinney, W. (2018). *Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython*.
- VanderPlas, J. (2016). *Python Data Science Handbook*.
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*.
- Hunter, J.D. (2007). *Matplotlib: A 2D Graphics Environment*. Computing in Science & Engineering.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research.
- Official Python Documentation: https://docs.python.org
- Seaborn Documentation: https://seaborn.pydata.org/
- Statsmodels Documentation: https://www.statsmodels.org/


## 📝 Glosario de Términos y Fórmulas

| **Término**                            | **Definición**                                                                                              |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Media (μ)                              | μ = (1/n) ∑ᵢ xᵢ  # promedio de todos los valores.                                                             |
| Varianza (σ²)                          | σ² = (1/n) ∑ᵢ (xᵢ - μ)²  # medida de dispersión de los valores respecto a la media.                          |
| Desviación estándar (σ)                | σ = √[σ²]  # raíz cuadrada de la varianza.                                                                  |
| Z-score                                | z = (x - μ) / σ  # cuántas desviaciones estándar se aleja el valor x de la media.                           |
| Percentil                              | Valor por debajo del cual cae un p% de las observaciones.                                                   |
| Intervalo de confianza (95%)           | IC = x̄ ± 1.96 * (σ / √n)  # rango donde se espera que esté la media poblacional con 95% de confianza.      |
| Regresión lineal (OLS)                 | ŷ = β₀ + β₁ x  # modelo de regresión donde ŷ es el valor predicho.                                         |
| Error estándar de β₁                   | SE(β₁) = σ / √(∑ (xᵢ - x̄)²)  # precisión de la estimación de la pendiente.                                  |
| Coeficiente de correlación (r)         | r = Cov(X,Y) / (σ_X σ_Y)  # fuerza y dirección de la relación lineal entre X y Y.                           |
| Coeficiente de determinación (R²)      | R² = 1 - SSE/SST  # proporción de varianza de Y explicada por el modelo.                                    |
| train_test_split                       | Función de scikit-learn para separar datos en conjuntos de entrenamiento y prueba.                          |
| KNeighborsClassifier                   | Algoritmo de clasificación que asigna la clase según la mayoría de los k vecinos más cercanos.               |

