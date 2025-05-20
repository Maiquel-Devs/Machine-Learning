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
