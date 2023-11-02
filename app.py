import streamlit as st
import re
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np
from textblob import TextBlob

# Load the pre-trained model and tokenizer
model = tf.keras.models.load_model("modelfin.h5",compile=False)
tokenizer = Tokenizer()
tokenizer.word_index = np.load("word_index2.npy", allow_pickle=True).item()

# Define a list of basic stopwords
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"]

def preprocess_text(text):
    # Preprocess the input text as done during model training
    text = text.lower()
    text = re.sub(r"(@[A-Za-z0-9]+)|(\#[A-Za-z0-9]+)|(<Emoji:.*>)|(pic\.twitter\.com\/.*)", " ", text)
    text = ' '.join(re.sub("([^0-9A-Za-z \t])", " ", text).split())

    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=140)

    return padded_sequence

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)
    return prediction

# Streamlit UI
st.title("Sentiment Analysis by Anushka Kadam")

st.write("This app checks the sentiment of the given text input using TextBlob and a pre-trained model.")
st.write("TextBlob provides a quick analysis, and the model provides more in-depth analysis.")

user_input = st.text_area("Enter your text here:")

if st.button("Check Sentiment"):
    if user_input:
        # TextBlob Analysis
        textblob_analysis = TextBlob(user_input)
        textblob_sentiment = textblob_analysis.sentiment.polarity

        st.write("TextBlob Analysis:")
        if textblob_sentiment > 0:
            st.success("Positive")
        elif textblob_sentiment < 0:
            st.error("Negative")
        else:
            st.info("Neutral")

        # Model Analysis
        model_prediction = predict_sentiment(user_input)
        if model_prediction >= 0.5:
            st.write("Model Analysis:")
            st.error("Negative")
        else:
            st.write("Model Analysis:")
            st.success("Positive or Neutral")
    else:
        st.warning("Please enter text for analysis.")
