# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:46:03 2025

@author: apkun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === Load and Annotate ===
df = pd.read_csv("csgo_round_snapshots.csv")

# Assign round_id based on score changes
round_ids = []
current_round = 0
prev_ct_score = 0
prev_t_score = 0

for idx, row in df.iterrows():
    ct_score = row['ct_score']
    t_score = row['t_score']

    # Detect either a new round (score increase) or a new match (score reset)
    if (ct_score < prev_ct_score) or (t_score < prev_t_score):
        # Game reset
        current_round += 1
    elif (ct_score > prev_ct_score) or (t_score > prev_t_score):
        # Round increment within same match
        current_round += 1

    round_ids.append(current_round)
    prev_ct_score = ct_score
    prev_t_score = t_score

df['round_id'] = round_ids


# === Prepare Features and Labels ===
X = df.drop(columns=['round_winner', 'round_id'])
y = df['round_winner']
y_classes = y.map({'CT': 0, 'T': 1})  # Needed for metrics

# One-hot encode targets for training
y_encoded = pd.get_dummies(y).values

# Preprocess numeric and categorical columns
categorical_cols = ['map']
numerical_cols = X.select_dtypes(include=[np.number, bool]).columns.difference(categorical_cols)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

X_preprocessed = preprocessor.fit_transform(X)

# === Store Round IDs and Time Left Before Splitting ===
round_ids_array = df['round_id'].values
time_left_array = df['time_left'].values

# === Train/Test Split ===
X_train, X_test, y_train, y_test, y_classes_train, y_classes_test, round_ids_train, round_ids_test, time_left_train, time_left_test = train_test_split(
    X_preprocessed, y_encoded, y_classes, round_ids_array, time_left_array, test_size=0.2, random_state=42
)

# === Build and Train Neural Network ===
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_preprocessed.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')  # 2-class prediction
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=64)

# === Evaluate ===
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# === Predictions ===
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# === Confusion Matrix ===
cm = confusion_matrix(y_classes_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["CT", "T"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# === Classification Report ===
print("\nClassification Report:")
print(classification_report(y_classes_test, y_pred_classes, target_names=["CT", "T"]))

# === Plot Training History ===
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# === Export Snapshot Predictions with Round ID ===
X_test_df = pd.DataFrame(X_test.toarray() if hasattr(X_test, "toarray") else X_test)
X_test_df['true_class'] = y_classes_test.values
X_test_df['predicted_class'] = y_pred_classes
X_test_df['confidence_CT'] = y_pred_probs[:, 0]
X_test_df['confidence_T'] = y_pred_probs[:, 1]
X_test_df['round_id'] = round_ids_test
X_test_df['time_left'] = time_left_test
X_test_df.to_csv("snapshot_predictions_with_round_id.csv", index=False)

# === Enhancement 1: Accuracy per Round ===
per_round = X_test_df.groupby('round_id').agg({
    'true_class': 'first',
    'predicted_class': lambda x: x.mode()[0],  # Majority vote per round
})
per_round['correct'] = per_round['true_class'] == per_round['predicted_class']
round_level_accuracy = per_round['correct'].mean()
print(f"\n🔁 Per-Round Majority Vote Accuracy: {round_level_accuracy:.4f}")

# === Enhancement 2: Accuracy vs. Time Left (Binned) ===
X_test_df['time_bin'] = pd.cut(X_test_df['time_left'], bins=[0, 30, 60, 90, 120, 150, 180], labels=["0-30", "30-60", "60-90", "90-120", "120-150", "150-180"])

# Correct prediction accuracy per time bin
time_bin_accuracy = X_test_df.groupby('time_bin').apply(lambda df: (df['true_class'] == df['predicted_class']).mean())

# Incorrect prediction rate per time bin
time_bin_error = X_test_df.groupby('time_bin').apply(lambda df: (df['true_class'] != df['predicted_class']).mean())

# Plot correct predictions
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
time_bin_accuracy.plot(kind='bar', color='teal')
plt.title("✅ Accuracy by Time Left Bin")
plt.ylabel("Accuracy")
plt.xlabel("Time Left (seconds)")
plt.xticks(rotation=45)

# Plot incorrect predictions
plt.subplot(1, 2, 2)
time_bin_error.plot(kind='bar', color='salmon')
plt.title("❌ Error Rate by Time Left Bin")
plt.ylabel("Error Rate")
plt.xlabel("Time Left (seconds)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# === Enhancement 3: Earliest Accurate Snapshot per Round ===
first_correct_by_round = X_test_df[X_test_df['true_class'] == X_test_df['predicted_class']].groupby('round_id')['time_left'].max()
plt.hist(first_correct_by_round, bins=20, color='lightgreen', edgecolor='black')
plt.title("Time Left at First Correct Prediction (Per Round)")
plt.xlabel("Time Left (seconds)")
plt.ylabel("Rounds")
plt.tight_layout()
plt.show()

# === Notes for Further Enhancements ===
"""
📦 Now includes:
- Per-round prediction accuracy via majority vote
- Time-based accuracy and error rate analysis
- CSV logging for external investigation

🔍 Next possible steps:
- Group early vs. late snapshot reliability
- Integrate SHAP/LIME for model explanation
- Try RNN/LSTM to model rounds sequentially
"""
