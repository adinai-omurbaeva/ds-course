import pandas as pd

def preprocess_data():
    data = pd.read_csv("train.csv")
    data = data.drop(columns=['Name'])
    data = data.dropna()
    data = pd.get_dummies(data, drop_first=True)

    X = data.drop("Survived", axis=1)
    y = data["Survived"]
    preprocessed_data = pd.concat([X, y], axis=1)
    preprocessed_data.to_csv("data\\preprocessed.csv", index=False)

if __name__ == "__main__":
    preprocess_data()