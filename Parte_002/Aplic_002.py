# Aplic_001.py

from sklearn.linear_model import Perceptron
import numpy as np


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])


y = np.array([0, 0, 0, 1])

# Cria o modelo Perceptron 
model = Perceptron(max_iter=1000, eta0=0.1, random_state=42)   

# max_iter :  Quantas vezes o modelo treina (iterações)
# eta0 : Controla a velocidade dos ajustes dos pesos (taxa de aprendizado)
# random_state : 	Garante que os resultados sejam reproduzíveis    --> 42 é um numero padrão que nos usamos

# Treina o modelo  --> É onde envia os dados
model.fit(X, y)

# Testa o modelo
for xi in X:
    print(f"Entrada: {xi}, Saída: {model.predict([xi])[0]}")
