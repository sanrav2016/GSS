import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load the data
data = pd.read_csv("njdata1.csv")

# Drop rows with missing q_status or relevant columns
data = data.dropna(subset=["q_status", "q_year", "mw1"])

# Convert years to numerical if needed
data["q_year"] = pd.to_numeric(data["q_year"], errors="coerce")
data["prop_year"] = pd.to_numeric(data["prop_year"], errors="coerce")

# Define features and priority score
data["priority_score"] = data["mw1"] / (data["q_year"] + 1e-5)

# Features and target
features = ["q_year", "prop_year", "mw1", "type_clean", "service", "IA_status_clean", "county_1", "utility"]
X = pd.get_dummies(data[features], drop_first=True)  # Encode categorical variables
y = data["priority_score"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(random_state=42, n_estimators=300, max_depth=20))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Add predictions to the original dataset
data["predicted_priority"] = pipeline.predict(X)

# Sort tasks by predicted priority
priority_queue = data.sort_values(by="predicted_priority", ascending=False)

# Display top 10 tasks
print(priority_queue[["q_year", "mw1", "q_status", "predicted_priority"]].head(10))

#print(priority_queue[["q_year", "mw1", "q_status", "predicted_priority"]].all)