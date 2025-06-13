
# Python for Data Science

<img src="https://raw.githubusercontent.com/luis-fernandezt/Python-for-Data-Science/refs/heads/main/5%20An%C3%A1lisis%20de%20datos%20de%20Python/Ex_Files_Python_Data_Analysis/chapter4/pp.show().JPG" width="216" height="260">

Resumen de apuntes de Python para ciencia de datos utilizando Jupyter Notebook y Visual Studio Code.

---

## 📦 Librerías utilizadas

### Instalación con pip:

```bash
pip install pandas matplotlib seaborn numpy statsmodels scikit-learn dataclasses getweather
```

### Librerías importadas:

- `import pandas as pd` → **Manipulación y análisis de datos estructurados**
- `import numpy as np` → **Operaciones matemáticas y manejo eficiente de arrays**
- `import matplotlib.pyplot as plt` → **Visualización de datos en gráficos estáticos**
- `import seaborn as sns` → **Visualización estadística sobre matplotlib con estilos atractivos**
- `import statsmodels.api as sm` → **Modelos estadísticos como regresiones, pruebas y estimaciones**
- `from statsmodels.formula.api import ols` → **Regresión lineal con notación tipo fórmula**
- `from sklearn.model_selection import train_test_split` → **Separar datos en conjuntos de entrenamiento y prueba**
- `from sklearn.neighbors import KNeighborsClassifier` → **Modelo K-vecinos más cercanos para clasificación**
- `from datetime import date, time, datetime, timedelta` → **Manejo de fechas y tiempos**
- `import math` → **Funciones matemáticas básicas como seno, coseno, raíz cuadrada**
- `import os` / `from os import path` → **Interacción con el sistema operativo y rutas de archivos**
- `import shutil` / `from shutil import make_archive` → **Manejo de archivos y compresión de carpetas**
- `from zipfile import ZipFile` → **Lectura y escritura de archivos .zip**
- `import calendar` → **Calendarios, días de la semana, etc.**
- `import json` / `import urllib.request` → **Interacción con APIs y manejo de datos en formato JSON**
- `import collections` → **Estructuras como diccionarios ordenados, contadores y tuplas con nombre**
- `@dataclass` / `namedtuple` → **Estructuras de datos tipo objeto (POPO)**
- `import itertools` → **Herramientas para crear iteradores eficientes y combinatorios**
- `import getweather` → **Librería personalizada o externa para obtener datos del clima**

---

## 🛠️ Comandos y Sintaxis Útil

> Ordenados de menor a mayor complejidad y agrupados por funcionalidad.

### 📘 Entrada / Salida y estructuras básicas

```python
print("Hello World")  # salida simple en consola
mylist = [0, 1, "two", 3.2, False]  # lista con tipos mixtos
mydict = {"one": 1, "two": 2}  # diccionario clave:valor
del mylist  # elimina variable
del mydict  # elimina diccionario
```

### 📂 Archivos

```python
file = open("file.txt", "w+")  # escribir (sobrescribe si existe)
file = open("file.txt", "a+")  # agregar contenido al final
file = open("file.txt", "r")   # leer archivo
file.readlines()  # devuelve una lista con líneas del archivo
file.close()  # cierra el archivo para liberar recursos
```

### 📐 Tipos de datos y operaciones comunes

```python
len(data)  # número de elementos
type(data)  # tipo del objeto
sorted(data)  # devuelve una nueva lista ordenada
data.sort()  # ordena en sitio (modifica el original)
data.append("x")  # agrega un elemento
data.remove("x")  # elimina un elemento específico
data.insert(0, "x")  # inserta elemento en índice 0
data.extend(["a", "b"])  # agrega múltiples elementos
```

### 🔍 Pandas: limpieza y exploración de datos

```python
pd.read_csv("data.csv")  # lee archivo CSV
data.head()  # primeras 5 filas
data.tail()  # últimas 5 filas
data.info()  # resumen general del DataFrame
data.shape  # tupla (filas, columnas)
data.columns  # nombres de columnas
data.dtypes  # tipos de datos por columna
data.index  # índice del DataFrame
data.isna().sum()  # cantidad de NA por columna
data.dropna()  # elimina filas con NA
data.duplicated().sum()  # filas duplicadas
data["col"].value_counts(normalize=True)  # proporción de cada valor
data["col"].dtype  # tipo de dato de la columna
```

### 🔎 Pandas: filtrado y agrupación

```python
data["Year"] == 2025  # expresión booleana
data.loc[data["Year"] == 2025, :]  # filtra filas por año
data.groupby("Year").count()  # agrupación por año
data.sort_values("Count", ascending=False)  # ordena por columna
data.query("col > 10")  # filtra usando expresiones tipo SQL
```

### 🧪 Estadística básica

```python
np.mean(data["value"])  # media
np.min(data)  # mínimo
np.max(data)  # máximo
np.nanmin(data)  # mínimo ignorando NaN
np.nanmax(data)  # máximo ignorando NaN
np.isnan(data)  # identifica valores NaN
np.percentile(data, 2.5)  # percentil 2.5
np.percentile(data, 97.5)  # percentil 97.5
```

### 📊 Visualización

```python
data.plot.barh(x="Year", y="Count")  # gráfico de barras horizontal
sns.barplot(x="col1", y="col2", data=data, ci=False)  # barra con error estándar
sns.countplot(x="col", data=data)  # cuenta ocurrencias por categoría
sns.histplot(data=data, x="col", hue="group")  # histograma agrupado
sns.pairplot(data)  # matriz de gráficos entre pares
plt.hist(data["col"])  # histograma básico
plt.scatter(x=data["x"], y=data["y"], s=5)  # dispersión
plt.xlabel("x-axis")  # etiqueta eje x
plt.ylabel("y-axis")  # etiqueta eje y
plt.legend(["name1", "name2"])  # leyenda
plt.xlim(0, 100)  # límites eje x
plt.axhline(0)  # línea horizontal en y=0
plt.show()  # muestra el gráfico
pp.imshow() # muestra una imagen
pp.show() # muestra imagen
```

### 📈 Modelado

```python
X_train, X_test, y_train, y_test = train_test_split(X, y)  # divide los datos
knn = KNeighborsClassifier(n_neighbors=3)  # instancia modelo KNN
knn.fit(X_train, y_train)  # entrena modelo
knn.predict(X_test)  # predice con modelo
knn.score(X_test, y_test)  # exactitud

model = ols(formula="value1 ~ value2", data=data).fit()  # regresión lineal
model.summary()  # tabla resumen del modelo
sm.qqplot(model.resid, line='s')  # gráfico QQ para residuos
```

### 🧭 Fechas y tiempos

```python
datetime.now()  # fecha y hora actual
date.today()  # solo fecha actual
datetime.now().strftime("%Y-%m-%d")  # formateo personalizado
datetime.now() + timedelta(days=10)  # suma de días
datetime.now() - timedelta(weeks=1)  # resta de semanas
```

### 🌀 Control de flujo

```python
for x in range(10):  # bucle for
    print(x)

while x < 5:  # bucle while
    print(x)
    x += 1

days = ["Mon", "Tue", "Wed"]
for i, d in enumerate(days):  # enumerar con índice
    print(i, d)

if x > 0:  # condicionales
    print("Positive")
elif x == 0:
    print("Zero")
else:
    print("Negative")
```

---

## 🌐 Webpages y recursos en línea

- https://docs.python.org/3.6/library/datetime.html#strftime-and-strptime-behavior
- https://docs.python.org/es/3/library/json.html
- https://www.kaggle.com/
- https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
- https://html.onlineviewer.net/

---

## 📚 Glosario

| **Término / Fórmula**            | **Definición**                                                                 |
|----------------------------------|--------------------------------------------------------------------------------|
| `Media:` \( \mu = \frac{1}{n} \sum x_i \)     | Promedio aritmético de un conjunto de datos.                               |
| `Varianza:` \( \sigma^2 = \frac{1}{n} \sum (x_i - \mu)^2 \) | Mide la dispersión de los datos.                    |
| `Desviación estándar:` \( \sigma = \sqrt{\sigma^2} \)      | Raíz cuadrada de la varianza.                          |
| `Error estándar:` \( SE = \frac{\sigma}{\sqrt{n}} \)       | Precisión de la media muestral.                         |
| `Z-score:` \( z = \frac{x - \mu}{\sigma} \)   | Número de desviaciones estándar que un dato se aleja de la media.          |
| `IC 95%:` \( \bar{x} \pm 1.96 \cdot SE \)      | Intervalo de confianza para la media con 95% de seguridad.                 |
| `Regresión lineal:` \( y = \beta_0 + \beta_1 x + \epsilon \) | Relación lineal entre variable independiente y dependiente.              |
| `Correlación:` \( r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}} \) | Asociación entre dos variables. |
| `linear regression:`              | \( Y = \beta_0 + \beta_1 X + \epsilon \) → modelo de relación lineal     |
| `R² (R-squared):`                 | Proporción de varianza explicada por el modelo                                |
| `central limit theorem:`          | La media muestral se distribuye normal si n es suficientemente grande         |
| `type I error (α):`               | Rechazar la hipótesis nula cuando es verdadera (falso positivo)               |
| `type II error (β):`              | No rechazar la hipótesis nula cuando es falsa (falso negativo)                |
| `p-value:`                        | Probabilidad de obtener un resultado igual o más extremo, dado H₀ verdadera   |
| `one-tailed test:`                | Prueba estadística con hipótesis direccional (mayor o menor)                  |
| `two-tailed test:`                | Prueba estadística donde interesa desviación en ambas direcciones             |
| `hypothesis:`                     | Suposición que puede ser probada con datos                                    |


**Autor:**  
Luis Fernández  
Jupyter & Python Notes — 2025  