from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import pandas as pd

# Load dataset
df = pd.read_csv("diabetes.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = RandomForestClassifier(n_estimators=200, max_depth=6)
model.fit(X_train_scaled, y_train)

# Accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"🔥 Accuracy: {accuracy * 100:.2f}%")

# Save
pickle.dump(model, open("models/diabetes_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("✅ Model & scaler saved")