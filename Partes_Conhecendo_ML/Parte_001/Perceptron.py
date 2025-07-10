# Importa a biblioteca NumPy, que ajuda a trabalhar com vetores e matrizes
import numpy as np

# Define a função de ativação do perceptron (função degrau)
def step_function(x):
    # Se o valor for maior ou igual a 0, retorna 1. Caso contrário, retorna 0.
    return 1 if x >= 0 else 0

# Criação da classe Perceptron (modelo de aprendizado de máquina)
class Perceptron:

    # Método que é executado quando um objeto do tipo Perceptron é criado
    def __init__(self, input_size, learning_rate=0.1):
        # Cria uma lista de pesos, começando com zeros, com o mesmo tamanho das entradas
        self.weights = np.zeros(input_size)
        # Inicializa o viés (bias) com zero
        self.bias = 0
        # Define a taxa de aprendizado (o quanto os pesos serão ajustados a cada erro)
        self.lr = learning_rate

    # Função que recebe uma entrada e calcula a saída do perceptron
    def predict(self, x):
        # Calcula o valor da combinação linear dos pesos com a entrada + bias
        z = np.dot(self.weights, x) + self.bias
        # Aplica a função degrau ao resultado e retorna 0 ou 1
        return step_function(z)

    # Função para treinar o perceptron com os dados de entrada (X) e saídas esperadas (y)
    def train(self, X, y, epochs=10):
        # Repete o processo de aprendizado várias vezes (epochs)
        for _ in range(epochs):
            # Para cada entrada (xi) e respectiva saída esperada (target)
            for xi, target in zip(X, y):
                # Calcula a saída do perceptron com os pesos atuais
                prediction = self.predict(xi)
                # Calcula o erro (diferença entre saída esperada e saída calculada)
                error = target - prediction
                # Ajusta os pesos com base no erro, taxa de aprendizado e entrada
                self.weights += self.lr * error * xi
                # Ajusta o bias com base no erro e taxa de aprendizado
                self.bias += self.lr * error

# Define as entradas para o problema da porta lógica AND
X = np.array([
    [0, 0],  # Entrada 1
    [0, 1],  # Entrada 2
    [1, 0],  # Entrada 3
    [1, 1]   # Entrada 4
])

# Define as saídas esperadas para a porta AND
# A saída só é 1 quando as duas entradas são 1
y = np.array([0, 0, 0, 1])

# Cria o perceptron com duas entradas (x1 e x2)
p = Perceptron(input_size=2)

# Treina o perceptron com os dados de entrada (X) e as saídas esperadas (y)
p.train(X, y)

# Testa o perceptron após o treinamento com todas as entradas
for xi in X:
    # Imprime a entrada e a saída gerada pelo perceptron
    print(f"Entrada: {xi}, Saída: {p.predict(xi)}")

