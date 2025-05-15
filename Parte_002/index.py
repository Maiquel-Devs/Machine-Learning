from sklearn.linear_model import Perceptron
import numpy as np

entradas = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

saidas_esperadas = np.array([0, 0, 0, 1])

# Cria o modelo de Perceptron
model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

# Treina o modelo com os dados
model.fit(entradas, saidas_esperadas)

# Testa as mesmas entradas usadas no treinamento
print("Testando entradas originais:")
for entrada in entradas:
    previsao = model.predict([entrada])[0]
    print(f"Entrada: {entrada}, Saída prevista: {previsao}")

# Testa novas entradas para verificar o aprendizado
novas_entradas = np.array([
    [0.9, 0.9],
    [0.2, 0.8],
    [0.8, 0.1],
    [0.5, 0.5],
    [1, 1],
    [0, 0]
])

print("\nTestando novas entradas:")
for entrada in novas_entradas:
    previsao = model.predict([entrada])[0]
    print(f"Entrada: {entrada}, Saída prevista: {previsao}")
