from sklearn.linear_model import Perceptron
import numpy as np

                                        
# --- Perceptron 1: operação AND ---
entrada_01 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  

saida_01 = np.array([0, 0, 0, 1])  

modelo_01 = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

modelo_01.fit(entrada_01, saida_01)

previsao_01 = modelo_01.predict(entrada_01)
print("Saída do Perceptron 1 (AND):", previsao_01)

                                       
# --- Perceptron 2: operação NOT aplicada à saída do Perceptron 1 ---
entrada_02 = previsao_01.reshape(-1, 1)  # previsao_01 pertence a Perceptron 1 

saida_02 = 1 - previsao_01  # operação NOT (inverte os valores)

modelo_02 = Perceptron(max_iter=1000, eta0=0.1, random_state=42)

modelo_02.fit(entrada_02, saida_02)

previsao_02 = modelo_02.predict(entrada_02)
print("Saída do Perceptron 2 (NOT da saída do P1):", previsao_02)


# Função reshape() serve para transforma o vetor em coluna. Isso ocorre porque  o Perceptron do sklearn exige que os dados de entrada (X) tenham duas dimensões: linhas e colunas. 


# A comunicação é basicamente pegar a previsão/resposta do perceptron 1 (salvando em uma variável), e depois jogar essa variável no perceptron 2.