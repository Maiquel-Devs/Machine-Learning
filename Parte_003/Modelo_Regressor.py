from sklearn.linear_model import LinearRegression
import numpy as np

# Exemplo simples: quanto mais horas você estuda, maior sua nota
horas_estudo = np.array([[1], [2], [3], [4], [5]])  # Entrada (X)
nota = np.array([2, 4, 6, 8, 10])                  # Saída (y)

modelo = LinearRegression()
modelo.fit(horas_estudo, nota)

# Prevendo a nota para 6 horas de estudo
previsao = modelo.predict([[6]])
print(f"Nota prevista para 6 horas de estudo: {previsao[0]:.2f}")



# Atráves dos Dados que voçê manda ele prever resultados 

# Você treina o modelo com dados reais, e depois ele aprende o padrão para prever novos valores.

# O modelo percebeu que a nota dobra o valor das horas (nota = 2 × horas).

# Com isso, ele consegue prever para qualquer número de horas, mesmo que não estivesse nos dados de treino.from sklearn.linear_model import LinearRegression
import numpy as np

# Exemplo simples: quanto mais horas você estuda, maior sua nota
horas_estudo = np.array([[1], [2], [3], [4], [5]])  # Entrada (X)
nota = np.array([2, 4, 6, 8, 10])                  # Saída (y)

modelo = LinearRegression()
modelo.fit(horas_estudo, nota)

# Prevendo a nota para 6 horas de estudo
previsao = modelo.predict([[6]])
print(f"Nota prevista para 6 horas de estudo: {previsao[0]:.2f}")



# Atráves dos Dados que voçê manda ele prever resultados 

# Você treina o modelo com dados reais, e depois ele aprende o padrão para prever novos valores.

# O modelo percebeu que a nota dobra o valor das horas (nota = 2 × horas).

# Com isso, ele consegue prever para qualquer número de horas, mesmo que não estivesse nos dados de treino.