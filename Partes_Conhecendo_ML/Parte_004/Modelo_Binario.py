from sklearn.linear_model import LogisticRegression
import numpy as np

# Dados de exemplo (idade)
X = np.array([[18], [22], [25], [30], [35], [40], [45], [50], [55], [60]])
# Classe: 0 = não comprou, 1 = comprou
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# Cria o modelo
modelo = LogisticRegression()

# Treina o modelo
modelo.fit(X, y)

# Faz previsões
idades_para_testar = np.array([[20], [28], [38], [52]])
previsoes = modelo.predict(idades_para_testar)

# Exibe os resultados
for idade, resultado in zip(idades_para_testar, previsoes):
    print(f"Idade {idade[0]} => {'Comprou' if resultado == 1 else 'Não comprou'}")


# O Modelo prever que ...

# Idades menores que 28-30 anos têm probabilidade abaixo de 0.5 → o modelo prevê que não compram.

# Idades maiores que 30 anos têm probabilidade acima de 0.5 → o modelo prevê que compram.

