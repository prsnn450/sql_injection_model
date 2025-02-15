import joblib

# Load the saved model
model = joblib.load("intrusion_detection_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Example: Predict on new data
new_text = ["select * from users where id  =  '1' or \.<\ union select 1,@@VERSION -- 1"]
new_text_tfidf = vectorizer.transform(new_text)  # Transform using the same TF-IDF vectorizer
prediction = model.predict(new_text_tfidf)

print("Prediction:", prediction)
