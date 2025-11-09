import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import joblib

# ---- Generate sample dataset ----
np.random.seed(42)
data_size = 200

sleep_duration = np.random.uniform(4, 10, data_size)
screen_time = np.random.uniform(0, 4, data_size)
caffeine = np.random.randint(0, 6, data_size)
stress_level = np.random.randint(1, 11, data_size)
physical_activity = np.random.randint(0, 180, data_size)
age = np.random.randint(18, 60, data_size)

# A simple logic for "sleep quality"
sleep_quality = (
    10 * sleep_duration
    - 2 * screen_time
    - 3 * caffeine
    - 1.5 * stress_level
    + 0.05 * physical_activity
    - 0.1 * age
    + np.random.normal(0, 5, data_size)
)
sleep_quality = np.clip(sleep_quality, 0, 100)

# ---- Create DataFrame ----
df = pd.DataFrame({
    'sleep_duration': sleep_duration,
    'screen_time': screen_time,
    'caffeine': caffeine,
    'stress_level': stress_level,
    'physical_activity': physical_activity,
    'age': age,
    'sleep_quality': sleep_quality
})

# ---- Train a model ----
X = df[['sleep_duration', 'screen_time', 'caffeine', 'stress_level', 'physical_activity', 'age']]
y = df['sleep_quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# ---- Evaluate ----
preds = model.predict(X_test)
print("R² Score:", r2_score(y_test, preds))

# ---- Save trained model ----
joblib.dump(model, 'sleep_quality_model.joblib')
print("✅ Model saved as 'sleep_quality_model.joblib'")
