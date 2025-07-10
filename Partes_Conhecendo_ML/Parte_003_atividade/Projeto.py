import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Dados fictícios: área, quartos, banheiros, preço
data = {
    'area': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    'quartos': [1, 2, 2, 3, 3, 4, 4, 4, 5, 5],
    'banheiros': [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
    'preco': [100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000]
}

df = pd.DataFrame(data)


x = df[['area', 'quartos', 'banheiros']]
y = df['preco']


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# Criando modelo de regressão linear
modelo = LinearRegression()

# Treinando Modelo
modelo.fit(X_train, y_train)  # --> 80% dos dados serão treinado 


# Fazendo Previsões de Teste
y_pred = modelo.predict(X_test)

print('MSE:', mean_squared_error(y_test, y_pred))
print('R²', r2_score(y_test, y_pred))


# Fazendo Previsões com Novas Entradas
nova_casa = [[100, 3, 2]]
previsao = modelo.predict(nova_casa)
print(f"Preço estimado: R${previsao[0]:,.2f}")


# Aqui nesse caso 80% dos dados são usados para treino e 20% para fazer previsões.

# Se você treinasse e testasse com os mesmos dados, o modelo "decoraria" os valores — mas não saberia prever novos casos. Separando assim, você testa o modelo com dados que ele nunca viu.

# test_size=0.2 --> Parametro que que define que 80% dos dados vão para treino e outros 20% para previsões.


# random_state=0 --> Garante que a divisão sempre seja a mesma toda vez que o código rodar


# mean_squared_error() --> Serve para corrigir erros , elevano ao quadrado