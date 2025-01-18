# test_model.py

import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved TF-IDF Vectorizer and Model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = load_model('drug_trafficking_model_with_augmentation_new.h5')

# Load test data (ensure you have a CSV file with comments to test)
test_comments_df = pd.read_csv('comments_test.csv', encoding='Windows-1252')  # Use appropriate encoding

# Preprocess the test comments using the loaded vectorizer
X_test = test_comments_df['comment'].values
X_test_tfidf = vectorizer.transform(X_test).toarray()  # Transform using the fitted vectorizer

# Make predictions using the loaded model
predictions = model.predict(X_test_tfidf)

# Convert predictions to binary labels (0 or 1)
predicted_labels = (predictions > 0.5).astype(int)

# Add predictions to the DataFrame for analysis
test_comments_df['predicted_label'] = predicted_labels

# Save the results to a new CSV file
test_comments_df.to_csv('predictions_augmentation_new.csv', index=False)

print("Predictions saved to predictions_augmentation_new.csv")