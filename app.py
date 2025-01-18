from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the trained model and TF-IDF vectorizer
try:
    model = load_model('drug_trafficking_model_with_augmentation_new.h5')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or vectorizer: {str(e)}")

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Allow non-ASCII characters in JSON responses

@app.route('/')
def home():
    return render_template('updated_app.html')  # Render the updated HTML template for input

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the comment from the POST request
        comment = request.form.get('comment', '').strip()  # Use get() to avoid KeyError
        
        if not comment:
            return jsonify({'error': 'Comment cannot be empty.'}), 400  # Bad request if empty
        
        logging.info(f"Received comment: {comment}")  # Log the received comment
        
        # Transform the comment using the loaded vectorizer
        comment_tfidf = vectorizer.transform([comment]).toarray()
        
        # Make prediction
        prediction = model.predict(comment_tfidf)
        
        # Convert prediction to binary (0 or 1)
        result = 1 if prediction[0][0] > 0.5 else 0
        
        logging.info(f"Prediction result: {result}")  # Log the prediction result
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        logging.error(f"Error occurred during prediction: {str(e)}")  # Log detailed error message
        return jsonify({'error': 'An error occurred during prediction. Please try again later.'}), 500  # Return error message with a 500 status code

if __name__ == '__main__':
    app.run(debug=True)  # Ensure debug=True is set for detailed error messages
