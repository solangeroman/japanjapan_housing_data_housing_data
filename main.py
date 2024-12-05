import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score, mean_absolute_error
from imblearn.under_sampling import NearMiss 
from imblearn.under_sampling import RandomUnderSampler

dataframe_code = pd.read_csv("prefecture_code.csv")
dataframe_code = dataframe_code.rename(columns={"Code": "Prefecture_ID", "EnName": "Prefecture"})
dataframe_code = dataframe_code.drop(columns=['JpName'])

dataframe_code

folder = "trade_prices"
dataframes = []

for archivo in os.listdir(folder):
    if archivo.endswith(".csv"):
        ruta_archivo = os.path.join(folder, archivo)
        df = pd.read_csv(ruta_archivo)
        
        df = df.iloc[:, 1:]
        
        dataframes.append(df)

df_final = pd.concat(dataframes, ignore_index=True)
df_final.index = range(1, len(df_final) + 1)

df_final

dataframe_final = pd.merge(df_final, dataframe_code, on="Prefecture", how="left")

dataframe_final.head(10)

dataframe_final['Year'] = pd.to_numeric(dataframe_final['Year'], errors='coerce')
ultimo_anio = dataframe_final['Year'].max()
df_ultimos_10_anios = dataframe_final[dataframe_final['Year'] >= (ultimo_anio - 10)]
tendencia_precio = df_ultimos_10_anios.groupby('Year')['TradePrice'].median().reset_index()
tendencia_precio

plt.figure(figsize=(8, 3))
plt.plot(tendencia_precio['Year'], tendencia_precio['TradePrice'], marker='o', linestyle='-', color='blue')
plt.title('Tendencia de precios de bienes inmuebles en Japón (Últimos 10 años)')
plt.xlabel('Año')
plt.ylabel('Precio de Transacción (Mediana)')
plt.grid(True)
plt.show()

dataframe_final['Location'] = dataframe_final['Prefecture'].apply(lambda x: 'Tokyo' if x == 'Tokyo' else 'Other')
estadisticas = dataframe_final.groupby('Location')['TradePrice'].describe()

# Imprimir estadísticas descriptivas
print(estadisticas)

tokyo_prices = dataframe_final[dataframe_final['Location'] == 'Tokyo']['TradePrice']
other_prices = dataframe_final[dataframe_final['Location'] == 'Other']['TradePrice']
t_stat, p_value = ttest_ind(tokyo_prices, other_prices, nan_policy='omit')

print(f"Estadístico t: {t_stat}")
print(f"Valor p: {p_value}")


# Visualizar con un gráfico de cajas (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Location', y='TradePrice', data=dataframe_final)
plt.title('Comparación de precios de bienes inmuebles: Tokio vs Áreas Locales')
plt.xlabel('Ubicación')
plt.ylabel('Precio de Transacción')
plt.grid(True)
plt.show()

# Verificar duplicados
duplicados = dataframe_final.duplicated()

# Contar el número total de duplicados
num_duplicados = duplicados.sum()
print(f"Total de filas duplicadas: {num_duplicados}")

# Mostrar ejemplos de duplicados (si existen)
if num_duplicados > 0:
    print("Ejemplos de filas duplicadas:")
    print(dataframe_final[duplicados].head())
else:
    print("No se encontraron duplicados.")

df_final_sin_duplicados = dataframe_final.drop_duplicates()
print(f"Total de filas después de eliminar duplicados: {len(df_final_sin_duplicados)}")

df_final_sin_duplicados

# Seleccionar las columnas relevantes
caracteristicas = ['Area', 'Year', 'Prefecture_ID', 'UnitPrice']  # Agregar más columnas relevantes según corresponda
df_final_sin_duplicados = df_final_sin_duplicados[caracteristicas + ['TradePrice']].dropna()  # Eliminar filas con valores faltantes

X = df_final_sin_duplicados[caracteristicas]
y = df_final_sin_duplicados['TradePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




