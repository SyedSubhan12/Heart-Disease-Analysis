from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import joblib
from preprocessing import X, y

# Load the dataset
df = pd.read_csv("D:/heart_disease_prediction/data/transformed_heart_disease.csv")



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model
random_forest = RandomForestClassifier(random_state=42)

# Define the hyperparameters for GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and the best estimator
best_rf = grid_search.best_estimator_
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Evaluate the model
accuracy = best_rf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Print the classification report
cls_r = classification_report(y_test, best_rf.predict(X_test))
print(cls_r)

# Save the model
joblib.dump(best_rf, "best_random_forest_model.pkl")
print("Model saved successfully!")
