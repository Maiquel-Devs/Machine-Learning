import pandas as pd

# Carrega o arquivo CSV
df = pd.read_csv("XAUUSD_2004-2024.csv")

# Mostra as primeiras linhas do dataset
print(df.head())

# Informações gerais do dataset
print(df.info())
