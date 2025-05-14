from sklearn.linear_model import Perceptron
import numpy as np

# Dados de entrada da porta AND
entradas = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Resultados esperados (saída da porta AND)
saidas_esperadas = np.array([0, 0, 0, 1])

# Cria o modelo de Perceptron
modelo = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

# Treina o modelo com os dados
modelo.fit(entradas, saidas_esperadas)

# Testa o modelo com cada entrada
for entrada in entradas:
    previsao = modelo.predict([entrada])[0]
    print(f"Entrada: {entrada}, Saída: {previsao}")
