from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load the IMDB dataset and word index
max_features = 10000
maxlen = 200
(x_train, _), (_, _) = imdb.load_data(num_words=max_features)
word_index = imdb.get_word_index()

# Load LSTM model
lstm_model = load_model('model_lstm_final.h5')

# Function to preprocess text and predict sentiment
def predict_sentiment(model, text):
    words = text.split()
    words = [word_index[word] + 3 for word in words if word in word_index]
    words = pad_sequences([words], maxlen=maxlen)
    prediction = model.predict(words)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return prediction, sentiment

# Function to compute metrics and generate plots
def evaluate_model(model, x_test, y_test):
    y_pred = (model.predict(x_test) >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return accuracy, confusion, precision, recall

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction, sentiment = predict_sentiment(lstm_model, text)
    
    if sentiment == "Positive":
        image = "happy.jpg"
    else:
        image = "sad.jpg"
    
    return render_template('result.html', prediction=prediction, sentiment=sentiment, image=image)

@app.route('/evaluate')
def evaluate():
    # Split the data for evaluation
    _, (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_test = pad_sequences(x_test, maxlen=maxlen)
    
    # Evaluate LSTM model
    lstm_accuracy, lstm_confusion, lstm_precision, lstm_recall = evaluate_model(lstm_model, x_test, y_test)
    
    # Plot graphs
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.bar(['LSTM'], [lstm_accuracy], color=['blue'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    
    plt.subplot(2, 2, 2)
    plt.bar(['LSTM'], [1-lstm_accuracy], color=['blue'])
    plt.title('Error')
    plt.ylabel('Error')
    
    plt.subplot(2, 2, 3)
    sns.heatmap(lstm_confusion, annot=True, cmap='Blues', fmt='g', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('LSTM Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, f'Precision: {lstm_precision:.2f}\nRecall: {lstm_recall:.2f}', fontsize=14, ha='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('evaluation.png')
    
    return render_template('evaluation.html')

if __name__ == '__main__':
    app.run(debug=True)
