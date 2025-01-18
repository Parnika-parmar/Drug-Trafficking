import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved TF-IDF Vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load the saved Keras model
model = load_model('drug_trafficking_model_with_augmentation_new.h5')

# Function to predict drug-related statements
def predict_drug_related_statement(statement):
    # Transform the input statement using the loaded vectorizer
    statement_tfidf = vectorizer.transform([statement]).toarray()  # Transform to TF-IDF format

    # Make prediction using the loaded model
    prediction = model.predict(statement_tfidf)

    # Convert prediction to binary label (0 or 1)
    predicted_label = (prediction > 0.5).astype(int)

    return predicted_label[0][0]  # Return the predicted label

# Example usage
if __name__ == "__main__":
    print("Drug-Related Statement Prediction")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("Enter a statement to check if it's drug-related: ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        
        result = predict_drug_related_statement(user_input)
        
        if result == 1:
            print("The statement is related to drugs.")
        else:
            print("The statement is not related to drugs.")