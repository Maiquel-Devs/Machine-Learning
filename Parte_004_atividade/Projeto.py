import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Criar base de dados fictícia
dados = {
    'horas_estudo': [2, 5, 1, 3, 6, 8, 4, 7, 2, 9],
    'faltas':       [5, 2, 6, 3, 1, 0, 4, 1, 6, 0],
    'aprovado':     [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]  # 1 = aprovado, 0 = reprovado
}

df = pd.DataFrame(dados)

# 2. Separar X (entradas) e y (saída)
X = df[['horas_estudo', 'faltas']]
y = df['aprovado']

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Treinar modelo de regressão logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# 5. Testar com novos dados
novos_alunos = pd.DataFrame({
    'horas_estudo': [1, 6, 3],
    'faltas': [6, 0, 2]
})

previsoes = modelo.predict(novos_alunos)

# 6. Mostrar resultados
for i, resultado in enumerate(previsoes):
    status = 'Aprovado ✅' if resultado == 1 else 'Reprovado ❌'
    print(f"Aluno {i+1}: {status}")



# Aqui os dados são respondidos com 0 ou 1.

# O Modelo prever se os novos alunos têm chance de serem aprovados.

# Parecido com o projeto anterior mas aqui os dados são respondidos com 0 e 1. 