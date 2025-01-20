import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import RandomOverSampler

def load_data(file_path):
    """Loads the dataset and returns a DataFrame."""
    return pd.read_csv(file_path)

def drop_columns(df, columns_to_drop):
    """Drops specified columns from the DataFrame."""
    return df.drop(columns=columns_to_drop, axis=1)

def handle_missing_values(df, placeholder="?", strategy='median'):
    """Replaces placeholders with 'Unknown' and handles missing values using the specified strategy."""
    df = df.replace(placeholder, 'Unknown')
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            if strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                raise ValueError("Invalid strategy. Choose 'median' or 'mode'.")
    return df

def encode_categorical_columns(df, categorical_columns):
    """Encodes categorical columns using LabelEncoder."""
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(str)
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders

def scale_numerical_columns(df, numerical_columns):
    """Scales numerical columns using StandardScaler."""
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler

def balance_dataset(X, y):
    """Balances the dataset using RandomOverSampler."""
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    return X_resampled, y_resampled

def save_transformed_data(df, output_file):
    """Saves the transformed data to a CSV file."""
    df.to_csv(output_file, index=False)

def define_features_and_target(df, target_column):
    """
    Defines X (features) and y (target) for machine learning.
    Drops the target column from features.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def transform_data(file_path, output_file):
    """Executes the full data transformation pipeline."""
    df = load_data(file_path)

    # Step 1: Drop unnecessary columns
    columns_to_drop = ['id', 'dataset']  # Adjust this list based on your dataset
    df = drop_columns(df, columns_to_drop)

    # Step 2: Handle missing or placeholder values
    df = handle_missing_values(df)

    # Step 3: Encode categorical features
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    df, label_encoders = encode_categorical_columns(df, categorical_columns)

    # Step 4: Scale numerical features
    numerical_columns = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
    df, scaler = scale_numerical_columns(df, numerical_columns)

    # Step 5: Define X (features) and y (target)
    target_column = 'num'  # Target column for predicting heart disease
    X, y = define_features_and_target(df, target_column)

    # Step 6: Balance the dataset
    X, y = balance_dataset(X, y)

    # Save the transformed dataset
    transformed_df = pd.concat([pd.DataFrame(X, columns=df.drop(columns=[target_column]).columns), pd.DataFrame(y, columns=[target_column])], axis=1)
    save_transformed_data(transformed_df, output_file)

    return X, y, label_encoders, scaler

file_path = 'D:/heart_disease_prediction/data/heart_disease_uci.csv'
output_file = 'transformed_heart_disease.csv'
X, y, label_encoders, scaler = transform_data(file_path, output_file)

print("Features (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())
