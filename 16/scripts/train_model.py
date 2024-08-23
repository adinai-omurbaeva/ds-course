import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
def load_data():
    data = pd.read_csv("train.csv")
    return data

def preprocess_data(data):
    data = data.dropna()
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    
    return X, y

def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier())
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    data = load_data()
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение моделей
    boosting_model = train_model(X_train, y_train)
    
    # Оценка моделей
    boosting_accuracy = evaluate_model(boosting_model, X_test, y_test)
    
    # Вывод результатов
    print(f"Gradient Boosting Accuracy: {boosting_accuracy:.4f}")