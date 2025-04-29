# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 16:03:38 2025

@author: apkun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === Load and Annotate ===
df = pd.read_csv("csgo_round_snapshots.csv")

# Assign round_id based on score changes or resets
round_ids = []
current_round = 0
prev_ct_score = 0
prev_t_score = 0

for idx, row in df.iterrows():
    ct_score = row['ct_score']
    t_score = row['t_score']
    if (ct_score < prev_ct_score) or (t_score < prev_t_score):
        current_round += 1
    elif (ct_score > prev_ct_score) or (t_score > prev_t_score):
        current_round += 1
    round_ids.append(current_round)
    prev_ct_score = ct_score
    prev_t_score = t_score

df['round_id'] = round_ids

# === Prepare Features and Labels ===
X = df.drop(columns=['round_winner', 'round_id'])
y = df['round_winner']
y_classes = y.map({'CT': 0, 'T': 1})
y_encoded = pd.get_dummies(y).values

categorical_cols = ['map']
numerical_cols = X.select_dtypes(include=[np.number, bool]).columns.difference(categorical_cols)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

X_preprocessed = preprocessor.fit_transform(X)

# Store round and time data
round_ids_array = df['round_id'].values
time_left_array = df['time_left'].values

# === Split dataset ===
X_train, X_test, y_train_enc, y_test_enc, y_classes_train, y_classes_test, round_ids_train, round_ids_test, time_left_train, time_left_test = train_test_split(
    X_preprocessed, y_encoded, y_classes, round_ids_array, time_left_array, test_size=0.2, random_state=42
)

# === Build Neural Network with Functional API (Fixes input error) ===
input_layer = Input(shape=(X_train.shape[1],))
x = Dense(128, activation='relu')(input_layer)
x = Dropout(0.3)(x)
embedding = Dense(64, activation='relu', name='embedding_layer')(x)
x = Dropout(0.3)(embedding)
output = Dense(2, activation='softmax')(x)

nn_model = Model(inputs=input_layer, outputs=output)
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train_enc, validation_split=0.2, epochs=10, batch_size=64)

# === Extract embeddings ===
embedding_model = Model(inputs=nn_model.input, outputs=nn_model.get_layer("embedding_layer").output)
X_train_embed = embedding_model.predict(X_train)
X_test_embed = embedding_model.predict(X_test)

# === Train Random Forest and SVM on embeddings ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
rf.fit(X_train_embed, y_classes_train)
svm.fit(X_train_embed, y_classes_train)

# === Ensemble: VotingClassifier ===
ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('svm', svm)
], voting='soft')
ensemble.fit(X_train_embed, y_classes_train)

# === Evaluate Ensemble ===
y_pred_ensemble = ensemble.predict(X_test_embed)
y_proba_ensemble = ensemble.predict_proba(X_test_embed)

print("\n🔮 Ensemble Accuracy:", accuracy_score(y_classes_test, y_pred_ensemble))
print("\n📋 Classification Report:")
print(classification_report(y_classes_test, y_pred_ensemble, target_names=["CT", "T"]))

# === Confusion Matrix ===
cm = confusion_matrix(y_classes_test, y_pred_ensemble)
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix - Ensemble")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0, 1], ["CT", "T"])
plt.yticks([0, 1], ["CT", "T"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.colorbar()
plt.show()

# === Export Snapshot Predictions with Round ID ===
X_test_df = pd.DataFrame(X_test.toarray() if hasattr(X_test, "toarray") else X_test)
X_test_df['true_class'] = y_classes_test.values
X_test_df['predicted_class'] = y_pred_ensemble
X_test_df['confidence_CT'] = y_proba_ensemble[:, 0]
X_test_df['confidence_T'] = y_proba_ensemble[:, 1]
X_test_df['round_id'] = round_ids_test
X_test_df['time_left'] = time_left_test
X_test_df.to_csv("snapshot_predictions_with_round_id.csv", index=False)

# === Per-Round Accuracy (Majority Vote) ===
per_round = X_test_df.groupby('round_id').agg({
    'true_class': 'first',
    'predicted_class': lambda x: x.mode()[0]
})
per_round['correct'] = per_round['true_class'] == per_round['predicted_class']
round_level_accuracy = per_round['correct'].mean()
print(f"\n🔁 Per-Round Majority Vote Accuracy: {round_level_accuracy:.4f}")

# === Accuracy vs. Time Left Bins ===
X_test_df['time_bin'] = pd.cut(X_test_df['time_left'], bins=[0, 30, 60, 90, 120, 150, 180], labels=["0-30", "30-60", "60-90", "90-120", "120-150", "150-180"])
time_bin_accuracy = X_test_df.groupby('time_bin').apply(lambda df: (df['true_class'] == df['predicted_class']).mean())
time_bin_error = X_test_df.groupby('time_bin').apply(lambda df: (df['true_class'] != df['predicted_class']).mean())

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
time_bin_accuracy.plot(kind='bar', color='teal')
plt.title("✅ Accuracy by Time Left Bin")
plt.ylabel("Accuracy")
plt.xlabel("Time Left (seconds)")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
time_bin_error.plot(kind='bar', color='salmon')
plt.title("❌ Error Rate by Time Left Bin")
plt.ylabel("Error Rate")
plt.xlabel("Time Left (seconds)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# === Earliest Correct Prediction per Round ===
first_correct_by_round = X_test_df[X_test_df['true_class'] == X_test_df['predicted_class']].groupby('round_id')['time_left'].max()
plt.hist(first_correct_by_round, bins=20, color='lightgreen', edgecolor='black')
plt.title("Time Left at First Correct Prediction (Per Round)")
plt.xlabel("Time Left (seconds)")
plt.ylabel("Rounds")
plt.tight_layout()
plt.show()

