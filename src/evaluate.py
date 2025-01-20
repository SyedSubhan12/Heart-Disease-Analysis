import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv("D:/heart_disease_prediction/data/transformed_heart_disease.csv")

# Load the saved model
model_path = "D:/heart_disease_prediction/models/heart_disease_voting_model.pkl"
loaded_model = joblib.load(model_path)

# Separate features and target
X = df.drop(columns=['num'])  # Replace 'num' with the actual target column name if different
y = df['num']

# Split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Evaluate the model
print("Evaluating the model...")
y_pred = loaded_model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print metrics
print(f"Model Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print(y_pred)

# Save the evaluation results
with open("model_evaluation_report.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy:.2f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{conf_matrix}\n\n")
    f.write("Classification Report:\n")
    f.write(class_report)

print("Evaluation results saved to 'model_evaluation_report.txt'.")
