import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved TF-IDF Vectorizer and Model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = load_model('drug_trafficking_model_with_augmentation_new.h5')  # Ensure the correct model file is used

def predict_drug_related_statement(statement):
    # Transform the input statement using the loaded vectorizer
    statement_tfidf = vectorizer.transform([statement]).toarray()  # Transform to TF-IDF format

    # Make prediction using the loaded model
    prediction = model.predict(statement_tfidf)

    # Convert prediction to binary label (0 or 1)
    predicted_label = (prediction > 0.5).astype(int)

    return predicted_label[0][0]  # Return the predicted label

if __name__ == "__main__":
    while True:
        # Input statement from user
        user_input = input("Enter a statement to check if it's drug-related (or type 'exit' to quit): ")
        
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        
        # Predict and display result
        result = predict_drug_related_statement(user_input)
        if result == 1:
            print("The statement is related to drugs.")
        else:
            print("The statement is not related to drugs.")