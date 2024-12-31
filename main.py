import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load the data
data = pd.read_csv("njdata1.csv")

# Drop rows with missing q_status
data = data.dropna(subset=["q_status"])

# Feature selection
features = [
    "q_year", "prop_year", "type_clean", "mw1", "county_1", "service", "IA_status_clean", "utility"
]
X = data[features]
y = data["q_status"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing
categorical_features = ["type_clean", "service", "IA_status_clean", "county_1", "utility"]
numerical_features = ["q_year", "prop_year", "mw1"]

# Preprocessing pipeline
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Define the model
rf_model = RandomForestClassifier(random_state=42)

# Create a pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", rf_model)
])

# Parameter tuning
param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_leaf": [1, 2, 5],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation
print("Best Parameters:", grid_search.best_params_)
print("Accuracy on Test Data:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

rf_model = best_model.named_steps["classifier"]

# Get feature names after preprocessing
categorical_features_encoded = best_model.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_features)
all_feature_names = np.hstack([numerical_features, categorical_features_encoded])

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame for easier interpretation
importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Display the top features
print(importance_df.head(10))

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.title("Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()