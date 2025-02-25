import joblib

# Load the saved Random Forest model
rf_model = joblib.load("random_forest_model.pkl")

# Load the saved Isolation Forest model
iso_forest = joblib.load("isolation_forest_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Example: Predict on new data
new_text = ["select * from users where id  =  '1' or \.<\ union select 1,@@VERSION -- 1"]
new_text_tfidf = vectorizer.transform(new_text)  # Transform using the same TF-IDF vectorizer

# Prediction using Random Forest
rf_prediction = rf_model.predict(new_text_tfidf)
print("Random Forest Prediction:", rf_prediction)

# Prediction using Isolation Forest
iso_prediction = iso_forest.predict(new_text_tfidf)
iso_prediction = 1 if iso_prediction[0] == -1 else 0  # Convert -1 (anomaly) to 1 (intrusion), 1 to 0 (normal)
print("Isolation Forest Prediction:", iso_prediction)
