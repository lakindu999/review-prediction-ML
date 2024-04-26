from flask import Flask, render_template, request
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    if text:
        prediction = predict_sentiment(text)
        sentiment = sentiment_label(prediction)
        return render_template('index.html', sentiment=sentiment)
    else:
        return render_template('index.html', message="Please enter some text before predicting.")

if __name__ == '__main__':
    app.run(debug=True)
