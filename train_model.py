import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

data = pd.read_csv("data/students.csv")

print(data)

X = data[["study_hours", "attendance", "previous_score"]]

y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

prediction = model.predict([[6, 85, 75]])

print("Prediction:", prediction)
