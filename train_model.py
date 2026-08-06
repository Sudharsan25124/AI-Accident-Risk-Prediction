import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load Dataset
data = pd.read_csv("accident_data.csv")

# Convert text to numbers
le = LabelEncoder()

data["Weather"] = le.fit_transform(data["Weather"])
data["Traffic"] = le.fit_transform(data["Traffic"])
data["Risk"] = le.fit_transform(data["Risk"])

# Input and Output
X = data[["Weather", "Speed", "Traffic"]]
y = data["Risk"]

# Train Model
model = RandomForestClassifier()
model.fit(X, y)

# Save Model
joblib.dump(model, "accident_model.pkl")

print("Model Trained Successfully!")