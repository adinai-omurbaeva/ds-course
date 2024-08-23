from train_model import preprocess_data, train_model, evaluate_model
import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка данных из файла preprocessed.csv
data = pd.read_csv("data/preprocessed.csv")
X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели градиентного бустинга
model = train_model(X_train, y_train)
accuracy = evaluate_model(model, X_test, y_test)

# Запись результата в файл
with open("results/boosting_accuracy.txt", "w") as f:
    f.write(f"Gradient Boosting Accuracy: {accuracy:.4f}\n")