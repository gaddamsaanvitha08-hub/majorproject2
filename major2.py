
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
file_path = Path(__file__).parent / "sleep.csv"

if not file_path.exists():
    print("ERROR: sleep.csv was not found!")
    print("Please place sleep.csv in this folder:")
    print(Path(__file__).parent)
    raise SystemExit

data = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print(data.head())


data["Sleep Disorder"] = data["Sleep Disorder"].fillna("None")


features = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Stress Level",
    "Heart Rate",
    "Daily Steps"
]

X = data[features]
y = data["Sleep Disorder"].astype(str)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("\nModel training completed!")


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


new_person = pd.DataFrame([{
    "Age": 28,
    "Sleep Duration": 7.5,
    "Quality of Sleep": 8,
    "Stress Level": 4,
    "Heart Rate": 72,
    "Daily Steps": 8000
}])

prediction = model.predict(new_person)

print("\nPredicted Sleep Disorder:", prediction[0])
