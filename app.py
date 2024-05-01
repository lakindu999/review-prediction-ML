import streamlit as st
import pickle
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
with open('sentiment_analysis.pickle', 'rb') as file:
    model = pickle.load(file)

# Load the TF-IDF vectorizer
with open('tfidf_vectorizer.pickle', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '', text)  # Remove links
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    ps = PorterStemmer()
    text = ' '.join(ps.stem(word) for word in text.split())
    return text

# Prediction function
def predict_sentiment(text):
    # Preprocess input text
    preprocessed_text = preprocess_text(text)
    # Vectorize preprocessed text
    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])
    # Make prediction
    prediction = model.predict(vectorized_text)
    return prediction[0]  # Return the predicted sentiment label

# Function to convert sentiment label to string
def sentiment_label(prediction):
    return "Negative" if prediction == 1 else "Positive"

# Streamlit UI
st.title("User Review Sentiment Prediction System")
user_input = st.text_input("Enter your text:")

if st.button("Predict"):
    if user_input:
        prediction = predict_sentiment(user_input)
        sentiment = sentiment_label(prediction)
        label_color = "red" if sentiment == "Negative" else "green"
        colored_text = f'<span style="color: {label_color}; font-size: 20px;">{sentiment}</span>'
        final_result = f"That feedback is {colored_text}"
        st.markdown(final_result, unsafe_allow_html=True)
    else:
        st.write("Please Enter some text before predicting.")
