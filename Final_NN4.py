# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 20:08:28 2025

@author: apkun
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# === Load Dataset ===
print("📥 Loading dataset...")
df = pd.read_csv("csgo_round_snapshots.csv")

# Assign round_id based on score changes
print("🔁 Assigning round IDs...")
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
print("🧼 Preprocessing data...")
X = df.drop(columns=['round_winner', 'round_id'])
y = df['round_winner'].map({'CT': 0, 'T': 1})

categorical_cols = ['map']
numerical_cols = X.select_dtypes(include=[np.number, bool]).columns.difference(categorical_cols)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])

X_preprocessed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# === Define Models ===
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# === Train and Evaluate Models ===
results = {}
timings = {}

for name, model in models.items():
    print(f"\n⏳ Running model: {name}")
    start = time.time()

    print("  🔧 Training...")
    model.fit(X_train, y_train)

    print("  🔍 Predicting...")
    y_pred = model.predict(X_test)

    print("  📊 Evaluating...")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["CT", "T"])

    # Try to get prediction probabilities for ROC
    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_score = model.decision_function(X_test)

    end = time.time()

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "conf_matrix": cm,
        "report": report,
        "y_score": y_score
    }
    timings[name] = end - start
    print(f"  ✅ Done in {end - start:.2f} seconds.")

# === Plot Confusion Matrices ===
print("\n📈 Plotting confusion matrices...")
fig, axes = plt.subplots(1, len(results), figsize=(22, 5))
for ax, (name, res) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=res["conf_matrix"], display_labels=["CT", "T"])
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title(name)
plt.tight_layout()
plt.show()

# === Plot ROC Curves ===
print("\n📉 Plotting ROC curves...")
plt.figure(figsize=(10, 7))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_score"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curves.png")
plt.show()


# === Print Classification Reports ===
for name, res in results.items():
    print(f"\n📋 {name} Classification Report:")
    print(res["report"])

# === Print Timing Summary ===
print("\n⏱️ Model Processing Times:")
for name, seconds in timings.items():
    print(f"  - {name}: {seconds:.2f} seconds")

# === First Correct Prediction Timing Per Model (Bucketed) ===
print("\n📊 Plotting time buckets for first correct prediction per round (per model)...")

# Attach original data for reference (time_left and round_id)
X_with_meta = df[['round_id', 'time_left']].iloc[y_test.index].copy()
X_with_meta['true'] = y_test.values

bucket_size = 5  # seconds per bucket
max_time = df['time_left'].max()
bins = np.arange(0, max_time + bucket_size, bucket_size)
labels = [f"{i}-{i+bucket_size}" for i in bins[:-1]]

for name, res in results.items():
    print(f"📈 Processing model: {name}")
    preds = res['y_pred']
    X_with_meta['pred'] = preds

    # Find first correct prediction per round
    correct_times = []
    for rid, group in X_with_meta.groupby('round_id'):
        for _, row in group.iterrows():
            if row['true'] == row['pred']:
                correct_times.append(row['time_left'])
                break  # Only first correct prediction per round

    if correct_times:
        # Bin the times
        binned = pd.cut(correct_times, bins=bins, labels=labels, right=False)
        time_counts = binned.value_counts().sort_index()

        # Plot
        # Plot
        plt.figure(figsize=(10, 5))
        plt.bar(time_counts.index, time_counts.values)
        plt.xlabel("Time Remaining in Round (seconds, bucketed)")
        plt.ylabel("Number of Rounds")
        plt.title(f"{name}: Time Bucket of First Correct Prediction")
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f"first_correct_time_{name.replace(' ', '_').lower()}.png")
        plt.show()

    else:
        print(f"⚠️ No correct predictions found for model {name}")
# === Side-by-Side Bar Charts for Correct vs Incorrect First Predictions by Timestamp ===
print("\n📊 Plotting side-by-side correct vs incorrect first prediction timestamps...")

bucket_size = 5
max_time = df['time_left'].max()
bins = np.arange(0, max_time + bucket_size, bucket_size)
labels = [f"{i}-{i + bucket_size}" for i in bins[:-1]]

for name, res in results.items():
    print(f"📈 Processing model: {name}")
    preds = res['y_pred']
    X_with_meta['pred'] = preds

    correct_times = []
    incorrect_times = []

    for rid, group in X_with_meta.groupby('round_id'):
        first_row = group.iloc[0]
        if first_row['true'] == first_row['pred']:
            correct_times.append(first_row['time_left'])
        else:
            incorrect_times.append(first_row['time_left'])

    correct_binned = pd.cut(correct_times, bins=bins, labels=labels, right=False)
    incorrect_binned = pd.cut(incorrect_times, bins=bins, labels=labels, right=False)

    correct_counts = correct_binned.value_counts().sort_index()
    incorrect_counts = incorrect_binned.value_counts().sort_index()

    # Align counts to same index
    all_bins = sorted(set(correct_counts.index) | set(incorrect_counts.index))
    correct_vals = [correct_counts.get(bin_label, 0) for bin_label in labels]
    incorrect_vals = [incorrect_counts.get(bin_label, 0) for bin_label in labels]

    # Bar width & positions
    x = np.arange(len(labels))
    width = 0.4

    # Plot
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, correct_vals, width=width, label="Correct", color='green')
    plt.bar(x + width/2, incorrect_vals, width=width, label="Incorrect", color='red')
    plt.xticks(ticks=x, labels=labels, rotation=45)
    plt.xlabel("Time Remaining in Round (seconds, bucketed)")
    plt.ylabel("Number of Rounds")
    plt.title(f"{name}: First Prediction Timing — Correct vs Incorrect")
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"first_prediction_correct_vs_incorrect_{name.replace(' ', '_').lower()}.png")
    plt.show()

