import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from indicnlp.tokenize import indic_tokenize

app = Flask(__name__)

# Load the Telugu dataset
data = pd.read_csv("telugu_sentiment_data.csv")

# Drop rows with missing values
data = data.dropna()

# Split data into features and labels
X = data['words']
y = data['sentiment']

# Convert Telugu text data into numerical vectors
vectorizer = TfidfVectorizer(tokenizer=indic_tokenize.trivial_tokenize)
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Support Vector Machine (SVM) classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Function to classify input text
def classify_input(input_text):
    input_vector = vectorizer.transform([input_text])
    prediction = classifier.predict(input_vector)
    return prediction[0]

# Define your list of angry words here (Telugu)
angry_words_list = ['రోషం', 'కోపం', 'అసహ్యం']  # Add more angry words as needed

# Function to count and retrieve angry words
def count_angry_words(input_text):
    angry_word_count = 0
    angry_words = []
    for word in indic_tokenize.trivial_tokenize(input_text):
        if word in angry_words_list:
            angry_word_count += 1
            angry_words.append(word)
    return angry_word_count, angry_words

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for sentiment analysis
@app.route('/classify', methods=['POST'])
def classify():
    user_input = request.form['text']
    prediction = classify_input(user_input)
    angry_word_count, angry_words = count_angry_words(user_input)
    return render_template('result.html', prediction=prediction, angry_word_count=angry_word_count, angry_words=angry_words, input_text=user_input)

if __name__ == '__main__':
    app.run('0.0.0.0',debug=True)
