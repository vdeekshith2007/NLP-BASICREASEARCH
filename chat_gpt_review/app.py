from flask import Flask, render_template, request, jsonify
import pickle
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# this is to ensure required NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# Define preprocessing steps
custom_stopwords = {
    'chatgpt', 'ai', 'artificial intelligence', 'google', 'now', 'chatbot',
    'would', 'could', 'people', 'gpt', 'write', 'n', 'openai', 'using',
    'prompt', 'question', 'one', 'see', 'will', 'time', 'got', 'world'
}
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    filtered = [t for t in tokens if t not in custom_stopwords]
    lemmatized = [lemmatizer.lemmatize(w, pos='v') for w in filtered]
    return ' '.join(lemmatized)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     review = data.get('review', '')

#     if not review.strip():
#         return jsonify({'result': 'Please enter a review.'})

#     processed_review = preprocess(review)
#     vector = tfidf_vectorizer.transform([processed_review])
#     prediction = model.predict(vector)[0]

#     print("Input:", review)
#     print("Prediction:", prediction)

#     return jsonify({'result': prediction})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # parse JSON
    review = data.get('review', '')

    processed_review = preprocess(review)
    vector = tfidf_vectorizer.transform([processed_review])
    prediction = model.predict(vector)[0]
    
    return jsonify({'result': prediction.capitalize()})  # return JSON

if __name__ == '__main__':
    app.run(debug=True)
