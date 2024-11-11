# Install required libraries
# !pip install streamlit tensorflow numpy nltk

# ======================== Import Packages ===================================
import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# ======================== Load Model and Encoder ============================
# Load the pre-trained model
model = load_model('model.h5')

# Load the label encoder
import pickle
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# ======================== Helper Functions ==================================
def sentence_cleaning(sentence):
    """
    Cleans and preprocesses the input sentence for prediction.
    """
    stemmer = PorterStemmer()
    corpus = []
    text = re.sub("[^a-zA-Z]", " ", sentence)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    text = [stemmer.stem(word) for word in text if word not in stopwords]  # Remove stopwords and stem
    text = " ".join(text)
    corpus.append(text)
    one_hot_word = [one_hot(input_text=word, n=11000) for word in corpus]
    pad = pad_sequences(sequences=one_hot_word, maxlen=300, padding='pre')  # Pad the sequence
    return pad

def predict_emotion(sentence):
    """
    Predicts the emotion of the given sentence using the loaded LSTM model.
    """
    # Preprocess the input sentence
    cleaned_sentence = sentence_cleaning(sentence)

    # Predict using the LSTM model
    probabilities = model.predict(cleaned_sentence)  # Returns probabilities for all classes
    predicted_label_index = np.argmax(probabilities, axis=-1)[0]  # Get the index of the highest probability
    predicted_emotion = label_encoder.inverse_transform([predicted_label_index])[0]  # Decode the label
    probability = np.max(probabilities)  # Get the confidence score of the prediction

    return predicted_emotion, probability

# ======================== Streamlit App =====================================
st.title("Emotion Detection App")
st.write("This app detects human emotions such as Joy, Fear, Anger, Love, Sadness, and Surprise.")

# Input box for user text
user_input = st.text_input("Enter your text here:")

# Predict button
if st.button("Predict"):
    if user_input.strip():  # Ensure input is not empty
        emotion, confidence = predict_emotion(user_input)
        st.write(f"**Predicted Emotion:** {emotion}")
        st.write(f"**Confidence Score:** {confidence:.2f}")
    else:
        st.warning("Please enter some text before clicking Predict.")
