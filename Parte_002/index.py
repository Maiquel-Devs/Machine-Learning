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

# max_iter :  Quantas vezes o modelo treina (iterações)
# eta0 : Controla a velocidade dos ajustes dos pesos (taxa de aprendizado)
# random_state : 	Garante que os resultados sejam reproduzíveis    --> 42 é um numero padrão que nos usamos


# Treina o modelo com os dados  --> Aqui voçê está chamando a função pra enviar os dados.
model.fit(entradas, saidas_esperadas)


for entrada in entradas:
    previsao = model.predict([entrada])[0]
    print(f"Entrada: {entrada}, Saída: {previsao}")
