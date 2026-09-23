import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/students.csv")

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

plt.figure(figsize=(10, 6))

plot_tree(
    model,
    feature_names=["study_hours", "attendance", "previous_score"],
    class_names=["Fail", "Pass"],
    filled=True
)

plt.show()

tree_rules = export_text(
    model,
    feature_names=["study_hours", "attendance", "previous_score"]
)

print(tree_rules)

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved!")

predictions = model.predict(X_test)

print("Predictions:")
print(predictions)

print("Actual results:")
print(y_test.values)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

students = pd.DataFrame(
    [
        [1, 45, 30],
        [4, 70, 60],
        [8, 92, 90]
    ],
    columns=["study_hours", "attendance", "previous_score"]
)

predictions = model.predict(students)

print("New student predictions:")
print(predictions)
