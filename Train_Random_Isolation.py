import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score

nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("C:/Users/prsnn/Downloads/Modified_SQL_Dataset.csv")  # Replace with your file path
df.dropna(inplace=True)  # Drop missing values

# Text Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Apply preprocessing
df['clean_text'] = df['Query'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['Label'].values  

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------- Random Forest Model -----------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the Random Forest model
joblib.dump(rf_model, "random_forest_model.pkl")

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Evaluation Metrics for Random Forest
print("\n===== Random Forest Model =====")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-score:", f1_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# ----------------------- Isolation Forest Model (Anomaly Detection) -----------------------
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_forest.fit(X_train)

# Predicting anomalies (-1 for anomaly, 1 for normal)
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # Convert -1 (anomaly) to 1 (intrusion), 1 to 0 (normal)

# Save the Isolation Forest model
joblib.dump(iso_forest, "isolation_forest_model.pkl")

# Evaluation Metrics for Isolation Forest
print("\n===== Isolation Forest Model =====")
print("Accuracy:", accuracy_score(y_test, y_pred_iso))
print("Precision:", precision_score(y_test, y_pred_iso))
print("Recall:", recall_score(y_test, y_pred_iso))
print("F1-score:", f1_score(y_test, y_pred_iso))
print("\nClassification Report:\n", classification_report(y_test, y_pred_iso))
