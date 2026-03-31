import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 🔥 LOAD YOUR DATA DIRECTLY
df = pd.read_csv("data/processed/causal_variables_full_features.csv")

# Detect target column
target_col = None
for c in ['Accident_Severity', 'severity', 'target', 'label']:
    if c in df.columns:
        target_col = c
        break

if target_col is None:
    target_col = df.columns[-1]

print("Target column:", target_col)

# Split X, y
X = df.drop(columns=[target_col]).values
y = df[target_col].values

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train label dist:", np.bincount(y_train))

# Train RF
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print("\n==============================")
print("Random Forest Accuracy:", acc)
print("==============================")
