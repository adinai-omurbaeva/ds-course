import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import psycopg2
import numpy as np

def load_data():
    data = pd.read_csv("data/preprocessed.csv", encoding='utf-8-sig', sep=',')
    for col in data.columns:
        if data[col].dtype == object and set(data[col].unique()) <= {'False', 'True'}:
            data[col] = data[col].map({'False': False, 'True': True})
    
    print(data.dtypes)  
    return data

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model, model_name, scaler_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model_name": f"{model_name} with {scaler_name}",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    # Преобразование значений в стандартные типы Python
    metrics["accuracy"] = float(metrics["accuracy"])
    metrics["precision"] = float(metrics["precision"])
    metrics["recall"] = float(metrics["recall"])
    metrics["f1_score"] = float(metrics["f1_score"])

    return metrics

def save_metrics_to_db(metrics, conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.model_metrics (model_name, accuracy, precision, recall, f1_score)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (metrics["model_name"], metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"])
        )
    conn.commit()

if __name__ == "__main__":
    data = load_data()
    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Подключение к базе данных PostgreSQL
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )

    # Список скейлеров и моделей
    scalers = [
        (StandardScaler(), "Standard Scaler"),
        (MinMaxScaler(), "MinMax Scaler")
    ]

    models = [
        (LogisticRegression(), "Logistic Regression"),
        (GradientBoostingClassifier(), "Gradient Boosting"),
        (RandomForestClassifier(), "Random Forest")
    ]

    # Обучение моделей и сохранение метрик
    for scaler, scaler_name in scalers:
        for model, model_name in models:
            pipeline = Pipeline([
                ('scaler', scaler),
                ('model', model)
            ])
            metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, pipeline, model_name, scaler_name)
            save_metrics_to_db(metrics, conn)

    conn.close()
