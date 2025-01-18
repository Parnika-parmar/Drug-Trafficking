import pandas as pd
import joblib
import random
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from nltk.corpus import wordnet

# Ensure you have the WordNet corpus downloaded
nltk.download('wordnet')

# Load CSV files with specified encoding
comments_df = pd.read_csv('comments.csv', encoding='Windows-1252')  # Use Windows-1252 encoding

# Improved labeling function based on drug trafficking keywords
def label_drug_trafficking(comment):
    drug_related_keywords = [
        'drugs', 'trafficking', 'illegal', 'narcotic', 
        'smuggling', 'dealer', 'substance', 
        'addiction', 'cartel', 'distribution',
        'pain', 'pleasure', 'flee', 
        'blinded', 'troubles', 'criticize',
        'darkened', 'desire', 'error', 
        'laborious', 'narcotics', 'dope',
        'heroin', 'cocaine', 'marijuana'
    ]
    comment_lower = comment.lower()
    return 1 if any(keyword in comment_lower for keyword in drug_related_keywords) else 0

# Apply labeling function to comments
comments_df['label'] = comments_df['comment'].apply(label_drug_trafficking)

# Data Augmentation Function: Synonym Replacement with checks to avoid duplicates
def augment_text(text):
    words = text.split()
    new_words = words.copy()
    
    for i, word in enumerate(words):
        synonyms = wordnet.synsets(word)
        if synonyms:
            # Get a random synonym from the first synonym set
            synonym = random.choice(synonyms).lemmas()[0].name()
            if synonym != word:  # Only replace if different
                new_words[i] = synonym
            
    return " ".join(new_words)

# Augmenting the dataset by applying synonym replacement
augmented_comments = comments_df['comment'].apply(augment_text)
augmented_labels = comments_df['label']

# Combine original and augmented data
comments_df_augmented = pd.DataFrame({
    'comment': pd.concat([comments_df['comment'], augmented_comments]),
    'label': pd.concat([comments_df['label'], augmented_labels])
})

# Prepare features and labels
X = comments_df_augmented['comment'].values
y = comments_df_augmented['label'].values

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and save the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_val_tfidf = vectorizer.transform(X_val).toarray()  # Transform validation set

# Save the fitted vectorizer to a file
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  # Save to current directory

# Build and train the model with modifications
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train_tfidf.shape[1],)))  # Adjust input shape dynamically
model.add(BatchNormalization())
model.add(Dropout(0.5))  # Adjusted dropout rate for better regularization
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Implement early stopping and learning rate reduction on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

try:
    # Train the model with validation data and callbacks for early stopping and learning rate reduction
    history = model.fit(X_train_tfidf, y_train, epochs=20, validation_data=(X_val_tfidf, y_val), 
                        callbacks=[early_stopping, reduce_lr])
except Exception as e:
    print(f"An error occurred during model training: {e}")

# Save the trained model to an H5 file
model.save('drug_trafficking_model_with_augmentation_new.h5')
print("Model saved as drug_trafficking_model_with_augmentation_new.h5")

# Visualization of training history: Accuracy and Loss over epochs
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Show plots
plt.tight_layout()
plt.show()