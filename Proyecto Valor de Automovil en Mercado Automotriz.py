#!/usr/bin/env python
# coding: utf-8

# # Proyecto: Valor de Automóvil en Mercado Automotríz 
# 
# El servicio de venta de autos usados Rusty Bargain está desarrollando una aplicación para atraer nuevos clientes. Gracias a esa app, puedes averiguar rápidamente el valor de mercado de tu coche. Tienes acceso al historial: especificaciones técnicas, versiones de equipamiento y precios. Tienes que crear un modelo que determine el valor de mercado.
# A Rusty Bargain le interesa:
# - la calidad de la predicción;
# - la velocidad de la predicción;
# - el tiempo requerido para el entrenamiento

# ## Preparación de datos
# 
# Para preparar los datos para el modelo de predicción de valor de mercado de coches, primero cargamos los datos del archivo 'car_data.csv'. Luego, realizamos algunas operaciones básicas para explorar los datos y prepararlos para su procesamiento y modelado. 
# Este código carga los datos, muestra información básica sobre ellos (como las columnas y las primeras filas), convierte los nombres de las columnas a minúsculas para facilitar su manipulación y revisa la cantidad de valores faltantes y duplicados en el conjunto de datos. Estos son pasos importantes para la preparación inicial de los datos antes de realizar cualquier análisis o modelado. Aquí está el código hasta ahora:

# In[1]:


import numpy as np
import pandas as pd
import time

from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

df = pd.read_csv('/datasets/car_data.csv')
df.info()
print(df.columns)
print(df.head(10))
# bucle en los encabezados poniendo todo en minúsculas
new_col_names = []
for old_name in df.columns:
    #  todas las letras en minúsculas
    name_lowered = old_name.lower()
    new_col_names.append(name_lowered)
# Reemplaza los nombres anteriores por los nuevos
df.columns = new_col_names
print(df.columns)
# calcular el número de valores ausentes
print(df.isna().sum()) 
# contar duplicados explícitos
print(df.duplicated().sum()) 


# Los datos consisten en un conjunto de características de vehículos usados, como el precio, el tipo de vehículo, el año de registro, la caja de cambios, la potencia, el modelo, el kilometraje, el mes de registro, el tipo de combustible, la marca, el estado de reparación y algunas fechas y detalles de ubicación.
# 
# La información del conjunto de datos es la siguiente:
# 
# El conjunto de datos contiene 354,369 filas y 16 columnas.
# Hay varios tipos de datos presentes en el conjunto de datos, incluyendo datos numéricos (int64) y datos de texto (object).
# Algunas columnas tienen valores faltantes. Las columnas con valores faltantes son 'vehicletype', 'gearbox', 'model', 'fueltype' y 'notrepaired'. La columna 'notrepaired' es la que tiene más valores faltantes, con 71,154 registros.
# Hay 262 duplicados explícitos en el conjunto de datos.
# El siguiente paso será realizar el preprocesamiento de datos para manejar los valores faltantes y los duplicados, así como realizar la codificación de variables categóricas.
# 
# Después de eliminar los duplicados explícitos, ahora estamos imprimiendo los valores únicos para cada una de las columnas categóricas ('vehicletype', 'gearbox', 'model', 'fueltype' y 'notrepaired'), incluyendo los valores nulos. Esto nos ayudará a comprender mejor la distribución de los datos en estas columnas y nos permitirá decidir cómo manejar los valores faltantes durante el preprocesamiento de datos.
# 
# Por ejemplo, para la columna 'vehicletype', los valores únicos incluyen tipos de vehículos como 'bus' (autobús), 'convertible' (convertible), 'coupe', 'other' (otro), 'sedan' (sedán), 'small' (pequeño), 'suv' (SUV) y 'wagon' (familiar), además de valores nulos que indican que la información no está disponible.
# 
# Al ordenar los valores únicos, podemos obtener una idea clara de las categorías presentes en cada columna, lo que nos ayudará en las etapas posteriores del análisis y la preparación de datos.

# In[2]:


# eliminar duplicados explícitos
df = df.drop_duplicates().reset_index(drop=True)
# contar duplicados explícitos
print(df.duplicated().sum()) 

print("\nValores únicos de la columna 'vehicletype', incluidos los nulos y ordenados:")
print(sorted(df['vehicletype'].unique(), key=lambda x: str(x)))
print("\nValores únicos de la columna 'gearbox', incluidos los nulos y ordenados:")
print(sorted(df['gearbox'].unique(), key=lambda x: str(x)))
print("\nValores únicos de la columna 'model', incluidos los nulos y ordenados:")
print(sorted(df['model'].unique(), key=lambda x: str(x)))
print("\nValores únicos de la columna 'fueltype', incluidos los nulos y ordenados:")
print(sorted(df['fueltype'].unique(), key=lambda x: str(x)))
print("\nValores únicos de la columna 'notrepaired', incluidos los nulos y ordenados:")
print(sorted(df['notrepaired'].unique(), key=lambda x: str(x)))


# Después de eliminar los duplicados explícitos, hemos impreso los valores únicos para cada una de las columnas categóricas ('vehicletype', 'gearbox', 'model', 'fueltype' y 'notrepaired'), incluyendo los valores nulos, y los hemos ordenado alfabéticamente.
# 
# Aquí está la lista de valores únicos para cada columna:
# 
# - vehicletype : 'bus', 'convertible', 'coupe', 'nan', 'other', 'sedan', 'small', 'suv', 'wagon'
# 
# - gearbox : 'auto', 'manual', 'nan'
# 
# - model: '100', '145', '147', '156', '159', '1_reihe', '1er', '200', '2_reihe', '300c', '3_reihe', '3er', '4_reihe', '500', '5_reihe', '5er', '601', '6_reihe', '6er', '7er', '80', '850', '90', '900', '9000', '911', 'a1', 'a2', 'a3', 'a4', 'a5', 'a...
# 
# - fueltype': 'cng', 'electric', 'gasoline', 'hybrid', 'lpg', 'nan', 'other', 'petrol'
# notrepaired': 'nan', 'no', 'yes'
# 
# Estos resultados nos brindan una visión general de la diversidad de categorías presentes en cada columna y nos ayudarán en el proceso de preprocesamiento y análisis de datos posteriores.
# 
# El código acontinuación crea DataFrames con los valores únicos y sus recuentos para cada columna categórica. Luego, imprime estos recuentos en forma de tabla, seguido de estadísticas descriptivas y la matriz de correlación para el DataFrame principal df.

# In[3]:


# Crear un DataFrame con los valores únicos y sus conteos para cada columna
vehicletype_counts = df['vehicletype'].value_counts(dropna=False).reset_index()
vehicletype_counts.columns = ['Value', 'Count']

gearbox_counts = df['gearbox'].value_counts(dropna=False).reset_index()
gearbox_counts.columns = ['Value', 'Count']

model_counts = df['model'].value_counts(dropna=False).reset_index()
model_counts.columns = ['Value', 'Count']

fueltype_counts = df['fueltype'].value_counts(dropna=False).reset_index()
fueltype_counts.columns = ['Value', 'Count']

notrepaired_counts = df['notrepaired'].value_counts(dropna=False).reset_index()
notrepaired_counts.columns = ['Value', 'Count']

# Imprimir los conteos en forma de tabla
print("Tabla de conteo para la columna 'vehicletype':")
print(vehicletype_counts)

print("\nTabla de conteo para la columna 'gearbox':")
print(gearbox_counts)

print("\nTabla de conteo para la columna 'model':")
print(model_counts)

print("\nTabla de conteo para la columna 'fueltype':")
print(fueltype_counts)

print("\nTabla de conteo para la columna 'notrepaired':")
print(notrepaired_counts)

display(df.describe())
display(df.corr())


# Luego, se imprimen las estadísticas descriptivas y la matriz de correlación del DataFrame principal. Estas estadísticas proporcionan información útil sobre la distribución de los datos y las relaciones entre las diferentes variables.
# 
# El código realiza varias operaciones de preprocesamiento de datos:
# 
# Convierte las columnas 'datecrawled', 'datecreated' y 'lastseen' al tipo datetime para que puedan ser manipuladas más fácilmente. Calcula la edad del vehículo en años restando el año actual del año de registro del vehículo. Elimina las filas con valores negativos en la edad del vehículo, ya que estos no son válidos. Calcula la cantidad de días que un vehículo estuvo en línea para la venta restando la fecha de creación del anuncio de la fecha en que fue visto por última vez. Elimina columnas innecesarias para el modelo de predicción, como las relacionadas con fechas y códigos postales.
# 
# Después de estas operaciones, se muestran los tipos de datos actualizados, así como estadísticas descriptivas para las columnas 'vehicle_age' y 'days_online'. Finalmente, se muestra el conjunto de datos actualizado con las columnas preprocesadas.

# In[4]:


# Convertir las columnas 'datecrawled' y 'datecreated' a tipo datetime
df['datecrawled'] = pd.to_datetime(df['datecrawled'], format='%d/%m/%Y %H:%M')
df['datecreated'] = pd.to_datetime(df['datecreated'], format='%d/%m/%Y %H:%M')

# Convertir la columna 'lastseen' a tipo datetime
df['lastseen'] = pd.to_datetime(df['lastseen'], format='%d/%m/%Y %H:%M')

# Verificar los tipos de datos después de la conversión
print(df.dtypes)

# Calcular la edad del vehículo en años
current_year = pd.to_datetime('today').year
df['vehicle_age'] = current_year - df['registrationyear']

# Eliminar filas con valores negativos en la edad del vehículo
df = df[df['vehicle_age'] >= 0]

# Verificar estadísticas de la columna 'vehicle_age'
print(df['vehicle_age'].describe())

# Calcular la cantidad de días que un vehículo estuvo en línea para la venta
df['days_online'] = (df['lastseen'] - df['datecreated']).dt.days

# Verificar estadísticas de la columna 'days_online'
print(df['days_online'].describe())

# Eliminar columnas innecesarias para el modelo de predicción
columns_to_drop = ['datecrawled', 'datecreated', 'lastseen', 'numberofpictures']
df = df.drop(columns=columns_to_drop)

# Verificar el conjunto de datos después de eliminar columnas
print(df.head())


# - Tipos de datos actualizados: La columna 'datecrawled', 'datecreated' y 'lastseen' ahora son del tipo datetime64, lo que permite un manejo más conveniente de las fechas.
# 
# - Edad del vehículo: La edad promedio de los vehículos en el conjunto de datos es de aproximadamente 21 años, con una desviación estándar de aproximadamente 14 años. El vehículo más antiguo tiene 1024 años y el más reciente tiene 5 años.
# 
# - Días en línea para la venta: La cantidad promedio de días que un vehículo estuvo en línea para la venta es de aproximadamente 9 días, con una desviación estándar de aproximadamente 9 días. El vehículo que estuvo en línea por más tiempo estuvo disponible durante 759 días, mientras que el más corto estuvo disponible solo por 1 día.
# 
# - Conjunto de datos actualizado: Se eliminaron las columnas relacionadas con fechas ('datecrawled', 'datecreated', 'lastseen'), así como otras columnas no relevantes ('numberofpictures') para el modelo de predicción. Ahora, el conjunto de datos contiene las columnas preprocesadas y listas para ser utilizadas en la construcción del modelo de predicción.

# In[5]:


# Lista de columnas para las que queremos eliminar los valores NaN
columns_to_fillna = ['vehicletype', 'gearbox', 'model', 'fueltype', 'notrepaired']

# Completar los valores NaN con 'Unknown' para cada columna en la lista
for column in columns_to_fillna:
    df[column].fillna('Unknown', inplace=True)

# Imprimir el número de valores ausentes después de completar los NaN
print(df.isna().sum())
df.info()
# Verificar el conjunto de datos después de completar los valores NaN
print(df.head())


# Después de reemplazar los valores NaN en las columnas 'vehicletype', 'gearbox', 'model', 'fueltype' y 'notrepaired' por 'Unkown'. No hay valores faltantes en ninguna de las columnas restantes, lo que indica que se ha completado con éxito la eliminación de los valores NaN en las columnas especificadas. El conjunto de datos está listo para su uso en la construcción del modelo de predicción.

# In[6]:


display(df.describe())
display(df.corr())


# Los datos resumidos muestran que el precio promedio de los vehículos es de alrededor de 5,126 unidades monetarias, con un valor mínimo de 0 y un máximo de 20,000. La potencia promedio es de aproximadamente 120 caballos de fuerza, con una desviación estándar significativa de alrededor de 139. La antigüedad promedio de los vehículos es de aproximadamente 21 años, con un rango que va desde 6 a 114 años. La cantidad de días que un vehículo estuvo en línea para la venta tiene una media de alrededor de 9 días, con un máximo de 384 días.
# 
# En cuanto a las correlaciones, hay una correlación positiva moderada entre el precio y la potencia del vehículo, así como una correlación negativa moderada entre el precio y la antigüedad del vehículo. También se observa una correlación negativa débil entre el precio y el kilometraje, lo que sugiere que, en promedio, los vehículos con menor kilometraje tienden a tener un precio más alto. La antigüedad del vehículo y la cantidad de días en línea también tienen una correlación negativa débil, lo que indica que los vehículos más antiguos tienden a estar menos tiempo en línea para la venta.

# ## Entrenamiento del modelo 
# 
# ### Regresión Lineal
# 
# Acontinuación realizaremos varias tareas:
# 
# - Dividir los datos en conjuntos de entrenamiento y prueba, utilizando el 70% de los datos para entrenamiento y el 30% para pruebas.
# - Definir las columnas categóricas que serán procesadas.
# - Definit una transformación para las características categóricas, utilizando una tubería que primero imputa valores faltantes con "missing" y luego aplica codificación one-hot.
# - Combina todas las transformaciones usando ColumnTransformer.
# - Aplica la transformación a los datos de entrenamiento y prueba.
# - Entrena un modelo de regresión lineal utilizando los datos de entrenamiento preprocesados.
# - Realiza predicciones sobre el conjunto de prueba utilizando el modelo de Regresión Lineal.

# In[7]:


# Dividir los datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
X = df.drop(columns=['price'])  # Características
y = df['price']  # Etiquetas

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Definir las columnas categóricas
categorical_columns = ['vehicletype', 'gearbox', 'model', 'fueltype', 'brand', 'notrepaired']

# Definir la transformación para las características categóricas
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar las transformaciones
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_columns)
    ])

# Aplicar la transformación a los datos
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Entrenar el modelo de regresión lineal
linear_reg = LinearRegression()
linear_reg.fit(X_train_preprocessed, y_train)

# Predecir sobre el conjunto de prueba
y_pred_linear = linear_reg.predict(X_test_preprocessed)


# ### Modelo de Árbol de Decisión
# 
# Este fragmento de código entrena un modelo de regresión de árbol de decisión y luego realiza predicciones sobre el conjunto de prueba utilizando este modelo.
# 
# Aquí están los pasos realizados:
# 
# - Se instancia un modelo de árbol de decisión para regresión, utilizando el constructor DecisionTreeRegressor.
# 
# - Se entrena el modelo utilizando los datos de entrenamiento preprocesados (X_train_preprocessed y y_train) utilizando el método fit.
# 
# - Se realizan predicciones sobre el conjunto de prueba (X_test_preprocessed) utilizando el método predict, y las predicciones se almacenan en la variable y_pred_dt.
# 
# Este enfoque se basa en la estructura de árbol de decisiones para realizar predicciones. Cada nodo del árbol representa una característica, cada borde representa una decisión basada en esa característica, y cada hoja representa el valor de salida (en este caso, el precio del automóvil). El modelo aprende a hacer predicciones al dividir el espacio de características en regiones más pequeñas y hacer predicciones basadas en la media (o mediana) de los valores de salida en esas regiones.

# In[8]:


# Entrenar el modelo de árbol de decisión
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train_preprocessed, y_train)

# Predecir sobre el conjunto de prueba
y_pred_dt = decision_tree.predict(X_test_preprocessed)


# ### Modelo de Bosque Aleatorio
# 
# El siguiente código ajusta un modelo de bosque aleatorio para regresión con 30 árboles y una profundidad máxima de 5 niveles. Luego, realiza predicciones sobre el conjunto de prueba utilizando este modelo.
# 
# Resumen de los pasos:
# 
# - Se instancia un modelo de bosque aleatorio para regresión utilizando el constructor RandomForestRegressor.
# 
# - Se especifica el número de árboles (n_estimators) como 30 y la profundidad máxima (max_depth) como 5.
# 
# - Se entrena el modelo utilizando los datos de entrenamiento preprocesados (X_train_preprocessed y y_train) utilizando el método fit.
# 
# - Se realizan predicciones sobre el conjunto de prueba (X_test_preprocessed) utilizando el método predict, y las predicciones se almacenan en la variable y_pred_rf.
# 
# Limitar el número de árboles y la profundidad máxima puede ser útil para evitar el sobreajuste del modelo y mejorar su interpretabilidad.

# In[9]:


# Reducir el número de árboles y limitar la profundidad máxima
random_forest = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)
random_forest.fit(X_train_preprocessed, y_train)

# Predecir sobre el conjunto de prueba
y_pred_rf = random_forest.predict(X_test_preprocessed)


# ### Modelo LightGBM (Light Gradient-Boosting Machine)
# 
# Entrenamos un modelo de LightGBM para regresión y luego realizamos predicciones sobre el conjunto de prueba utilizando este modelo.
# 
# - Se instancia un modelo de LightGBM para regresión utilizando el constructor LGBMRegressor.
# 
# - Se establece una semilla aleatoria (random_state) para reproducibilidad.
# 
# - Se entrena el modelo utilizando los datos de entrenamiento preprocesados (X_train_preprocessed y y_train) utilizando el método fit.
# 
# - Se realizan predicciones sobre el conjunto de prueba (X_test_preprocessed) utilizando el método predict, y las predicciones se almacenan en la variable y_pred_lgbm.
# 
# LightGBM es un algoritmo de gradient boosting que a menudo se utiliza en problemas de regresión y clasificación debido a su eficiencia y capacidad para manejar grandes volúmenes de datos.

# In[10]:


# Entrenar el modelo de LightGBM
lgbm = LGBMRegressor(random_state=42)
lgbm.fit(X_train_preprocessed, y_train)

# Predecir sobre el conjunto de prueba
y_pred_lgbm = lgbm.predict(X_test_preprocessed)


# ### Modelo CatBoost
# 
# Entrenaremos un modelo de CatBoost para regresión y predicciones sobre el conjunto de prueba.
# 
# - Se instancia un modelo de CatBoost para regresión utilizando el constructor CatBoostRegressor.
# 
# - Se establece una semilla aleatoria (random_state) para reproducibilidad.
# 
# - Se entrena el modelo utilizando los datos de entrenamiento preprocesados (X_train_preprocessed y y_train) utilizando el método fit.
# 
# - Se realizan predicciones sobre el conjunto de prueba (X_test_preprocessed) utilizando el método predict, y las predicciones se almacenan en la variable y_pred_catboost.
# 
# CatBoost es una biblioteca de gradient boosting que también se utiliza comúnmente en problemas de regresión y clasificación. Ofrece un rendimiento excepcional y se destaca por su capacidad para manejar automáticamente variables categóricas sin necesidad de preprocesamiento adicional.

# In[11]:


# Entrenar el modelo de CatBoost
catboost = CatBoostRegressor(random_state=42, verbose=0)
catboost.fit(X_train_preprocessed, y_train)

# Predecir sobre el conjunto de prueba
y_pred_catboost = catboost.predict(X_test_preprocessed)


# ### Modelo SGDLinearRegression
# 
# La clase SGDLinearRegression implementa la regresión lineal utilizando el algoritmo de descenso de gradiente estocástico (SGD). Aquí está un resumen de los métodos y atributos de la clase:
# 
# - __init__: El constructor de la clase toma los siguientes parámetros:
#  - step_size: Tamaño del paso para la actualización de los pesos durante el descenso de gradiente.
#  - epochs: Número de épocas de entrenamiento.
#  - batch_size: Tamaño del lote para el descenso de gradiente estocástico.
#  - reg_weight: Peso de regularización para controlar el sobreajuste.
# - fit: Método para entrenar el modelo. Calcula los pesos de la regresión lineal utilizando el descenso de gradiente estocástico.
# - predict: Método para realizar predicciones utilizando el modelo entrenado.
# 
# La regularización L2 se aplica durante el entrenamiento para controlar el sobreajuste. La regularización se realiza agregando un término de penalización a la función de pérdida que penaliza los pesos grandes. Este término de penalización se calcula como el producto del peso de regularización y los pesos actuales del modelo. Se omite la penalización para el término de sesgo (w0).

# In[12]:


# Definir la clase SGDLinearRegression
class SGDLinearRegression:
    def __init__(self, step_size, epochs, batch_size, reg_weight):
        self.step_size = step_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_weight = reg_weight

    def fit(self, train_features, train_target):
        X = np.concatenate(
            (np.ones((train_features.shape[0], 1)), train_features.toarray()), axis=1
        )
        y = train_target
        w = np.zeros(X.shape[1])

        for _ in range(self.epochs):
            batches_count = X.shape[0] // self.batch_size
            for i in range(batches_count):
                begin = i * self.batch_size
                end = (i + 1) * self.batch_size
                X_batch = X[begin:end, :]
                y_batch = y[begin:end]

                gradient = (
                    2
                    * X_batch.T.dot(X_batch.dot(w) - y_batch)
                    / X_batch.shape[0]
                )
                
                # Regularización
                reg = 2 * w.copy()
                reg[0] = 0  # Se establece el elemento con índice cero en el vector reg a cero
                gradient += self.reg_weight * reg  # Agrega el sumando de regularización
                
                w -= self.step_size * gradient

        self.w = w[1:]
        self.w0 = w[0]

    def predict(self, test_features):
        return test_features.dot(self.w) + self.w0


# ## Análisis del modelo
# 
# Se han realizado las siguientes evaluaciones de los modelos:
# 
# - Regresión Lineal:
#  - RMSE: El error cuadrático medio (RMSE) del modelo de regresión lineal es rmse_linear.
#  - R^2: El coeficiente de determinación (R^2) del modelo de regresión lineal es r2_linear.
#  - Tiempos de Entrenamiento y Predicción: El tiempo de entrenamiento y predicción del modelo de regresión lineal son training_time_linear y prediction_time_linear respectivamente.
# 
# - Árbol de Decisión:
#  - RMSE: El RMSE del modelo de árbol de decisión es rmse_dt.
#  - R^2: El R^2 del modelo de árbol de decisión es r2_dt.
#  - Tiempos de Entrenamiento y Predicción: Los tiempos de entrenamiento y predicción del modelo de árbol de decisión son training_time_dt y prediction_time_dt respectivamente.
# 
# - Bosque Aleatorio:
#  - RMSE: El RMSE del modelo de bosque aleatorio reducido es rmse_rf.
#  - R^2: El R^2 del modelo de bosque aleatorio reducido es r2_rf.
#  - Tiempos de Entrenamiento y Predicción: Los tiempos de entrenamiento y predicción del modelo de bosque aleatorio son training_time_rf y prediction_time_rf respectivamente.
# 
# - LightGBM:
#  - RMSE: El RMSE del modelo de LightGBM es rmse_lgbm.
#  - R^2: El R^2 del modelo de LightGBM es r2_lgbm.
#  - Tiempos de Entrenamiento y Predicción: Los tiempos de entrenamiento y predicción del modelo de LightGBM son training_time_lgbm y prediction_time_lgbm respectivamente.
# 
# - CatBoost:
#  - RMSE: El RMSE del modelo de CatBoost es rmse_catboost.
#  - R^2: El R^2 del modelo de CatBoost es r2_catboost.
#  - Tiempos de Entrenamiento y Predicción: Los tiempos de entrenamiento y predicción del modelo de CatBoost son training_time_catboost y prediction_time_catboost respectivamente.

# In[13]:


# Calcular el RMSE (Root Mean Squared Error)
rmse_linear = mean_squared_error(y_test, y_pred_linear, squared=False)
print("\nRMSE del modelo de regresión lineal:", rmse_linear)
# Calcular R^2 para el modelo de regresión lineal
r2_linear = r2_score(y_test, y_pred_linear)
print("R^2 del modelo de regresión lineal:", r2_linear)
# Medición del tiempo de entrenamiento 
start_time = time.time()
linear_reg.fit(X_train_preprocessed, y_train)
training_time_linear = time.time() - start_time
# Medición del tiempo de predicción 
start_time = time.time()
y_pred_linear = linear_reg.predict(X_test_preprocessed)
prediction_time_linear = time.time() - start_time
# Imprimir los tiempos de entrenamiento y predicción 
print("Tiempo de Entrenamiento Regresión Lineal:", training_time_linear)
print("Tiempo de Predicción Regresión Lineal:", prediction_time_linear)

# Calcular el RMSE
rmse_dt = mean_squared_error(y_test, y_pred_dt, squared=False)
print("\nRMSE del modelo de árbol de decisión:", rmse_dt)
# Calcular R^2 para el modelo de árbol de decisión
r2_dt = r2_score(y_test, y_pred_dt)
print("R^2 del modelo de árbol de decisión:", r2_dt)
# Medición del tiempo de entrenamiento 
start_time = time.time()
decision_tree.fit(X_train_preprocessed, y_train)
training_time_dt = time.time() - start_time
# Medición del tiempo de predicción 
start_time = time.time()
y_pred_dt = decision_tree.predict(X_test_preprocessed)
prediction_time_dt = time.time() - start_time
# Imprimir los tiempos de entrenamiento y predicción
print("Tiempo de Entrenamiento Árbol de Decisión:", training_time_dt)
print("Tiempo de Predicción Árbol de Decisión:", prediction_time_dt)


# Calcular el RMSE
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print("\nRMSE del modelo de bosque aleatorio reducido:", rmse_rf)
# Calcular R^2 para el modelo de bosque aleatorio
r2_rf = r2_score(y_test, y_pred_rf)
print("R^2 del modelo de bosque aleatorio reducido:", r2_rf)
# Medición del tiempo de entrenamiento
start_time = time.time()
random_forest.fit(X_train_preprocessed, y_train)
training_time_rf = time.time() - start_time
# Medición del tiempo de predicción 
start_time = time.time()
y_pred_rf = random_forest.predict(X_test_preprocessed)
prediction_time_rf = time.time() - start_time
# Imprimir los tiempos de entrenamiento y predicción
print("Tiempo de Entrenamiento Bosque Aleatorio:", training_time_rf)
print("Tiempo de Predicción Bosque Aleatorio:", prediction_time_rf)

# Calcular el RMSE
rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm, squared=False)
print("\nRMSE del modelo de LightGBM:", rmse_lgbm)
# Calcular R^2 para el modelo de LightGBM
r2_lgbm = r2_score(y_test, y_pred_lgbm)
print("R^2 del modelo de LightGBM:", r2_lgbm)
# Medición del tiempo de entrenamiento
start_time = time.time()
lgbm.fit(X_train_preprocessed, y_train)
training_time_lgbm = time.time() - start_time
# Medición del tiempo de predicción
start_time = time.time()
y_pred_lgbm = lgbm.predict(X_test_preprocessed)
prediction_time_lgbm = time.time() - start_time
# Imprimir los tiempos de entrenamiento y predicción
print("Tiempo de Entrenamiento LightGBM:", training_time_lgbm)
print("Tiempo de Predicción LightGBM:", prediction_time_lgbm)


# Calcular el RMSE
rmse_catboost = mean_squared_error(y_test, y_pred_catboost, squared=False)
print("\nRMSE del modelo de CatBoost:", rmse_catboost)
# Calcular R^2 para el modelo de CatBoost
r2_catboost = r2_score(y_test, y_pred_catboost)
print("R^2 del modelo de CatBoost:", r2_catboost)
# Medición del tiempo de entrenamiento
start_time = time.time()
catboost.fit(X_train_preprocessed, y_train)
training_time_catboost = time.time() - start_time
# Medición del tiempo de predicción
start_time = time.time()
y_pred_catboost = catboost.predict(X_test_preprocessed)
prediction_time_catboost = time.time() - start_time
# Imprimir los tiempos de entrenamiento y predicción
print("Tiempo de Entrenamiento CatBoost:", training_time_catboost)
print("Tiempo de Predicción CatBoost:", prediction_time_catboost)


# - Regresión Lineal:
#  - RMSE: 3514.95
#  - R^2: 0.3960
#  - La regresión lineal es un modelo simple y fácil de interpretar. Sin embargo, los resultados muestran que su capacidad para capturar la variabilidad en los datos es limitada, como lo indica el valor moderado de R^2. Podría estar subestimando la complejidad de los datos, ya que no puede capturar relaciones no lineales.
# 
# - Árbol de Decisión:
#  - RMSE: 3396.41
#  - R^2: 0.4361
#  - Los árboles de decisión son modelos más flexibles que la regresión lineal y pueden capturar relaciones no lineales en los datos. Se destaca por su eficiencia y precisión. El resultado muestra una mejora en el RMSE y R^2 en comparación con la regresión lineal. Sin embargo, existe el riesgo de sobreajuste, especialmente en conjuntos de datos complejos.
# 
# - Bosque Aleatorio:
#  - RMSE: 3892.98
#  - R^2: 0.2591
#  - El bosque aleatorio es una extensión de los árboles de decisión que utiliza múltiples árboles para mejorar la precisión y reducir el sobreajuste. Sin embargo, en este caso, el modelo parece estar sobreajustando los datos, ya que el RMSE es mayor y el R^2 es menor que el modelo de árbol de decisión.
# 
# - LightGBM:
#  - RMSE: 3438.75
#  - R^2: 0.4219
#  - LightGBM es un algoritmo de aumento de gradiente que utiliza árboles de decisión como modelos base. En este caso, el modelo muestra un buen rendimiento en términos de RMSE y R^2, lo que indica que puede capturar de manera efectiva las relaciones en los datos.
# 
# - CatBoost:
#  - RMSE: 3370.62
#  - R^2: 0.4446
#  - CatBoost es otro algoritmo de aumento de gradiente que se destaca por su capacidad para manejar características categóricas sin la necesidad de codificación. Aunque el tiempo de entrenamiento es más largo que otros modelos, los resultados muestran un buen rendimiento en términos de RMSE y R^2.
# 
# En general, cada modelo tiene sus ventajas y desventajas. La elección del modelo dependerá de varios factores, incluida la complejidad de los datos, el tiempo de entrenamiento, la interpretabilidad y el rendimiento predictivo. Es importante considerar estos aspectos al seleccionar el modelo más adecuado para una aplicación específica.

# In[14]:


# Muestra en la pantalla los valores de peso de regularización y R2
print('\nRegularization:', 0.0)
model = SGDLinearRegression(0.01, 10, 100, 0.0)
model.fit(X_train_preprocessed, y_train)
pred_train = model.predict(X_train_preprocessed)
pred_test = model.predict(X_test_preprocessed)
print(r2_score(y_train, pred_train).round(5))
print(r2_score(y_test, pred_test).round(5))

# Calcular el RMSE para el modelo SGDLinearRegression
rmse_SGDLinearRegression = mean_squared_error(y_test, pred_test, squared=False)
print("RMSE del modelo SGDLinearRegression:", rmse_SGDLinearRegression)

print('\nRegularization:', 0.1)
model = SGDLinearRegression(0.01, 10, 100, 0.1)
model.fit(X_train_preprocessed, y_train)
pred_train = model.predict(X_train_preprocessed)
pred_test = model.predict(X_test_preprocessed)
print(r2_score(y_train, pred_train).round(5))
print(r2_score(y_test, pred_test).round(5))

# Calcular el RMSE para el modelo SGDLinearRegression
rmse_SGDLinearRegression = mean_squared_error(y_test, pred_test, squared=False)
print("RMSE del modelo SGDLinearRegression:", rmse_SGDLinearRegression)

print('\nRegularization:', 1.0)
model = SGDLinearRegression(0.01, 10, 100, 1.0)
model.fit(X_train_preprocessed, y_train)
pred_train = model.predict(X_train_preprocessed)
pred_test = model.predict(X_test_preprocessed)
print(r2_score(y_train, pred_train).round(5))
print(r2_score(y_test, pred_test).round(5))

# Calcular el RMSE para el modelo SGDLinearRegression
rmse_SGDLinearRegression = mean_squared_error(y_test, pred_test, squared=False)
print("RMSE del modelo SGDLinearRegression:", rmse_SGDLinearRegression)

print('\nRegularization:', 10.0)
model = SGDLinearRegression(0.01, 10, 100, 10.0)
model.fit(X_train_preprocessed, y_train)
pred_train = model.predict(X_train_preprocessed)
pred_test = model.predict(X_test_preprocessed)
print(r2_score(y_train, pred_train).round(5))
print(r2_score(y_test, pred_test).round(5))

# Calcular el RMSE para el modelo SGDLinearRegression
rmse_SGDLinearRegression = mean_squared_error(y_test, pred_test, squared=False)
print("RMSE del modelo SGDLinearRegression:", rmse_SGDLinearRegression)
# Entrenar el modelo SGDLinearRegression
start_time = time.time()
model.fit(X_train_preprocessed, y_train)
training_time_sgd = time.time() - start_time
# Predicción del tiempo para el modelo SGDLinearRegression
start_time = time.time()
pred_test_sgd = model.predict(X_test_preprocessed)
prediction_time_sgd = time.time() - start_time
# Imprimir los tiempos de entrenamiento y predicción
print("Tiempo de Entrenamiento SGDLinearRegression:", training_time_sgd)
print("Tiempo de Predicción SGDLinearRegression:", prediction_time_sgd)


# Estos resultados muestran que a medida que aumenta la regularización, el rendimiento del modelo disminuye tanto en términos de R^2 como de RMSE. Esto sugiere que la regularización está ayudando a controlar el sobreajuste del modelo. Sin embargo, incluso con la regularización más fuerte, el modelo aún muestra un rendimiento limitado en comparación con otros modelos más complejos. Además, el tiempo de entrenamiento del modelo SGDLinearRegression es considerablemente más largo que el de otros modelos, lo que puede deberse a la naturaleza iterativa del algoritmo de descenso de gradiente 

# In[ ]:


# Muestra en la pantalla los valores de peso de regularización y R2
print('\nRegularization:', 0.0)
model = SGDLinearRegression(0.1, 200, 2000, 0.0)
model.fit(X_train_preprocessed, y_train)
pred_train = model.predict(X_train_preprocessed)
pred_test = model.predict(X_test_preprocessed)
print(r2_score(y_train, pred_train).round(5))
print(r2_score(y_test, pred_test).round(5))

# Calcular el RMSE para el modelo SGDLinearRegression
rmse_SGDLinearRegression = mean_squared_error(y_test, pred_test, squared=False)
print("RMSE del modelo SGDLinearRegression:", rmse_SGDLinearRegression)
# Entrenar el modelo SGDLinearRegression
start_time = time.time()
model.fit(X_train_preprocessed, y_train)
training_time_sgd = time.time() - start_time
# Predicción del tiempo para el modelo SGDLinearRegression
start_time = time.time()
pred_test_sgd = model.predict(X_test_preprocessed)
prediction_time_sgd = time.time() - start_time
# Imprimir los tiempos de entrenamiento y predicción
print("Tiempo de Entrenamiento SGDLinearRegression:", training_time_sgd)
print("Tiempo de Predicción SGDLinearRegression:", prediction_time_sgd)


# Para el modelo SGDLinearRegression con un paso de aprendizaje de 0.1, 200 épocas, un tamaño de lote de 2000 y sin regularización, obtenemos los siguientes resultados:
# 
# - Regularización: 0.0
#  - R^2 en entrenamiento: 0.3967
#  - R^2 en prueba: 0.39435
#  - RMSE: 3519.9077798695885
# 
# Aunque el modelo mejora en términos de R^2 y RMSE en comparación con los casos anteriores sin regularización, el tiempo de entrenamiento aumenta significativamente a 62.3 segundos. Esto puede deberse al mayor número de épocas y al tamaño del lote más grande. La diferencia en el rendimiento del modelo no es sustancial, lo que sugiere que podría no ser necesario aumentar tanto la complejidad del modelo para obtener mejoras marginales en el rendimiento.
