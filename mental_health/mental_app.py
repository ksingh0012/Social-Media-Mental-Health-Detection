from flask import Flask, request, render_template, jsonify
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import os

# Initialize Flask app
app = Flask(__name__)

# Load models and vectorizers
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        raise RuntimeError(f"{file_path} not found.")

sentiment_model = load_model('model/sentiment/Logistic_Regression_best_model.pkl')
stress_model = load_model('model/stress/Logistic_Regression_stress_best_model.pkl')

sentiment_vectorizer = load_model('model/sentiment/tfidf_vectorizer_sentiment.pkl')
stress_vectorizer = load_model('model/stress/tfidf_vectorizer_stress.pkl')

# Initialize NLTK resources
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function to clean input text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+|[^a-zA-Z\s]', '', text)         # Remove mentions and special characters
    text = text.lower()                                  # Convert to lowercase
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '').strip() # reterive the text from the form and remove leading/trailing spaces

    if not text:
        return jsonify({'error': 'No text provided for prediction.'}), 400

    cleaned_text = clean_text(text)

    if not cleaned_text:
        return jsonify({'error': 'Input text is empty after cleaning.'}), 400

    try:
        # Generate predictions
        sentiment = "Positive" if sentiment_model.predict(sentiment_vectorizer.transform([cleaned_text]))[0] == 1 else "Negative"
        stress = "Stress" if stress_model.predict(stress_vectorizer.transform([cleaned_text]))[0] == 1 else "No Stress"

        # Determine mental state
        if stress == "No Stress" and sentiment == "Positive":
            mental_state = "Neutral/Healthy"
        elif stress == "Stress" and sentiment == "Positive":
            mental_state = "Stressed but Positive"
        elif stress == "Stress" and sentiment == "Negative":
            mental_state = "Critical State"
        elif stress == "No Stress" and sentiment == "Negative":
            mental_state = "Low Mood"
        else:
            mental_state = "Uncategorized"

        return jsonify({
            'sentiment_prediction': sentiment,
            'stress_prediction': stress,
            'mental_state': mental_state
        })

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
