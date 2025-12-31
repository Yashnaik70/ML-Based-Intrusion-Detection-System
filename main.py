# main.py
# Intrusion Detection System (IDS) using ML - Step 1: Load and Inspect Data

import pandas as pd

# Load the dataset
train_path = "Dataset/kdd_train.csv"
test_path = "Dataset/kdd_test.csv"

# Since NSL-KDD doesn't have headers, we'll add them manually later
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

print("✅ Training data loaded successfully!")
print("Training set shape:", df_train.shape)
print("Test set shape:", df_test.shape)

# Display first few rows
print("\nSample data:")
print(df_train.head())

# -----------------------------
# Step 2.1: Dataset Understanding
# -----------------------------

print("\n📌 Column Names:")
print(df_train.columns)

print("\n📌 Data Types:")
print(df_train.dtypes)

print("\n📌 Label Distribution:")
print(df_train['labels'].value_counts())

from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Step 2.2: Encode categorical columns (CORRECT WAY)
# -----------------------------

categorical_cols = ['protocol_type', 'service', 'flag']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    encoders[col] = le

print("\n✅ Categorical columns encoded successfully!")


# -----------------------------
# Step 2.3: Encode labels
# -----------------------------

df_train['labels'] = df_train['labels'].apply(
    lambda x: 0 if x == 'normal' else 1
)

df_test['labels'] = df_test['labels'].apply(
    lambda x: 0 if x == 'normal' else 1
)

print("✅ Labels encoded (0 = Normal, 1 = Attack)")

from sklearn.preprocessing import StandardScaler

# -----------------------------
# Step 2.4: Feature Scaling
# -----------------------------

X_train = df_train.drop('labels', axis=1)
y_train = df_train['labels']

X_test = df_test.drop('labels', axis=1)
y_test = df_test['labels']

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("✅ Data preprocessing completed!")
print("Total features:", X_train.shape[1])

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Step 3.2: Train Random Forest Model
# -----------------------------

print("\n🚀 Training Random Forest IDS model...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("✅ Model training completed!")

# -----------------------------
# Step 3.3: Model Prediction
# -----------------------------

y_pred = rf_model.predict(X_test)
print("✅ Predictions generated!")

# -----------------------------
# Step 3.4: Model Evaluation
# -----------------------------

accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Model Accuracy:", accuracy)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

# Save model and scaler
joblib.dump(rf_model, "ids_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "categorical_encoders.pkl")


print("💾 Model and scaler saved successfully!")
# -----------------------------