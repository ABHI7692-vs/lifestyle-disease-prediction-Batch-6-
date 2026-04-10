import pandas as pd
import numpy as np

np.random.seed(42)

data = []

for _ in range(5000):

    age = np.random.randint(18, 80)
    bmi = np.random.uniform(18, 40)

    smoking = np.random.randint(0, 4)      # 0–3
    alcohol = np.random.randint(0, 4)      # 0–3
    junk = np.random.randint(0, 3)         # 0–2
    exercise = np.random.randint(0, 4)     # 0–3
    sleep = np.random.randint(4, 10)

    gender = np.random.randint(0, 2)

    diabetes = np.random.randint(0, 2)
    hypertension = np.random.randint(0, 2)
    depression = np.random.randint(0, 2)

    # 🔥 SMART RISK FORMULA (IMPORTANT)
    risk_score = (
        0.03 * age +
        0.08 * bmi +
        0.15 * smoking +
        0.12 * alcohol +
        0.10 * junk -
        0.12 * exercise -
        0.05 * sleep +
        0.25 * diabetes +
        0.22 * hypertension +
        0.18 * depression
    )

    # Add noise
    risk_score += np.random.normal(0, 0.5)

    # Convert to binary target
    target = 1 if risk_score > 5 else 0

    data.append([
        age, bmi, alcohol, exercise, gender,
        junk, sleep, smoking,
        diabetes, hypertension, depression,
        target
    ])

columns = [
    "age", "bmi", "alcohol", "exercise", "gender",
    "junk", "sleep", "smoking",
    "diabetes", "hypertension", "depression",
    "target"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("health_data.csv", index=False)

print("✅ Dataset created: health_data.csv")