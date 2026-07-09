import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# ==========================================
# 1. LOAD DATA
# ==========================================
csv_filename = 'diabetes.csv'
target_col = 'Outcome'  

try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    print(f"❌ Error: Could not find '{csv_filename}'. Check the file name!")
    exit()

if target_col not in df.columns:
    print(f"❌ Error: Target column '{target_col}' not found.")
    print(f"Available columns are: {list(df.columns)}")
    exit()

# ==========================================
# 2. CLEAN & PREPROCESS DATA
# ==========================================
df = df.dropna()

X = df.drop(target_col, axis=1)
y = df[target_col]

# Scaling is highly important for medical data (like balancing age vs. insulin levels)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# 3. TRAIN MODEL
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Using Logistic Regression for this medical classification task
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 4. GET INTERNSHIP METRICS
# ==========================================
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("DISEASE PREDICTION MODEL REPORT")
print("="*50)
print(classification_report(y_test, predictions))

print(f"Overall Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
auc_score = roc_auc_score(y_test, probabilities)
print(f"ROC-AUC Score: {auc_score:.4f}")
print("="*50)