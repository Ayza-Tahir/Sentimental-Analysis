# -*- coding: utf-8 -*-
"""LogisticRegression.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CH4wkLzHJL9ySCtchJp_SAkWP33mf5di
"""

# isntall downgraded version of numpy
!pip install numpy

!pip install keras

!pip install tensorflow

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras.preprocessing import sequence
from sklearn.linear_model import LogisticRegression
# Set random seed for reproducibility
np.random.seed(7)

!pip install keras

# Load the dataset but only keep the top n words, zero the rest (vocabulary size as 5000)
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

print('---review---')

print(X_train[6])

print('---label---')

print(y_train[6])

word2id = imdb.get_word_index()

id2word = {i: word for word, i in word2id.items()}

print('---review with words---')

print([id2word.get(i,'') for i in X_train[6]])

print('---label---')
print(y_train[6])

print(word2id)

print(id2word)

# Maximum and minimum review length
print('Maximum review length: {}'.format(len(max((X_train + X_test), key=len))))
print('Minimum review length: {}'.format(len(min((X_train + X_test), key=len))))

# Pad sequences to ensure uniform length
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Flatten input sequences for Logistic Regression
X_train_flatten = np.reshape(X_train, (X_train.shape[0], -1))
X_test_flatten = np.reshape(X_test, (X_test.shape[0], -1))

from sklearn.metrics import log_loss

# Train the model over multiple epochs
train_accuracy_history = []
test_accuracy_history = []
loss_history = []  # To store loss values

epochs = 30

for epoch in range(epochs):
    # Train the model
    model.fit(X_train_flatten, y_train)

    # Evaluate training accuracy
    train_accuracy = model.score(X_train_flatten, y_train)
    train_accuracy_history.append(train_accuracy)

    # Evaluate testing accuracy
    test_accuracy = model.score(X_test_flatten, y_test)
    test_accuracy_history.append(test_accuracy)

    # Calculate and record loss
    y_pred_train = model.predict_proba(X_train_flatten)
    loss_train = log_loss(y_train, y_pred_train)
    loss_history.append(loss_train)

    print("Epoch {}: Training Accuracy = {:.4f}, Testing Accuracy = {:.4f}, Loss = {:.4f}".format(epoch+1, train_accuracy, test_accuracy, loss_train))

# Plotting the training and testing accuracies over epochs
plt.plot(range(1, epochs+1), train_accuracy_history, label='Training Accuracy')
plt.plot(range(1, epochs+1), test_accuracy_history, label='Testing Accuracy')
plt.title('Training and Testing Accuracies Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting the change in error (loss) over iterations
plt.plot(range(1, epochs+1), loss_history)
plt.title('Change in Error (Loss) Over Iterations')
plt.xlabel('Epoch')
plt.ylabel('Error (Loss)')
plt.show()

scores = model.evaluate(X_test,y_test, verbose = 0 )
print("accuracy: %2f" % (scores[1]*100))

!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
print(creds)
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()

!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

####code for sentimental analysis
@app.route('/'methods=['GET', 'POST'])
def home():

  return render_template("home.html")

#Mount your Google Drive

!mkdir -p drive
!google-drive-ocamlfuse drive

#After success run Drive Fuse progran, you can create a directory Sentiment _Analysis and access your drive
import os
os.mkdir("/content/drive/Sentiment_Analysis")
os.chdir("/content/drive/")
!ls

#Append your path
import sys
sys.path.append('/content/drive/Sentiment_Analysis')

#Now save the model in required directory
model.save('/content/drive/Sentiment_Analysis/sentiment_analysis_model_new.h5')
print("Saved model to disk")

os.chdir("/content/drive/Sentiment_Analysis")
!ls

#Code to load the saved model
model=load_model('/content/drive/Sentiment_Analysis/sentiment_analysis_model_new.h5')
print("Model Loaded")

from flask import Flask, render_template, flash,request,url_for
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model

IMAGE_FOLDER =os.path.join('static','img_pool')

app=Flask(__name__)

app.config['UPLOAD_FOLDER']

def init():
  global model,graph

  # load the pre-trained keras model
   model = load_model('sentiment_analysis.h5')
   graph= tf.get_default_graph()

   ######################################################Code for Sentiment Analysis

   @app.route('/',methods=['GET','POST'])

def home():

  return render_template("home.html")



   @app.route('sentimental_analysis_prediction', methods =['POST',"GET"])

def sent_anly_prediction ():
  if request.method=='POST':
    text = request.form['text']
    Sentiment = ''
    max_review_length = 500
    word_to_id = imdb.get_word_index()
    strip_special_chars = re.compiler("[^A-Za-z0-9 ]+")
    text = text.lower().replace("<br />"," ")
    text = re.sub(strip_special_chars,"",text.lower())

    words = text.split() # split string into a list
    x_test = [word_to_id[word] if (word in word_to_id and word_to_id[word] <= 20000 else 0 for word in word_to_id)]
    x_test = sequence.pad_sequences(x_test,maxlen=500)
    vector = np.array([x_test.flatten()])

    with graph.as_default():
      probability = model.predict(array([vector][0]))[0][0]
      class1 = model.predict_classes(array([vector][0]))[0][0]
      if class1 ==0 :
        sentiment = 'Negative'
        img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'Sad_Emoji.png')
      else:
        sentiment = 'Positive'
        img_filename = os.path.join(app.config['UPLOAD_FOLDER'],'Smiling_Emoji.png')
    return render_template('home.html',text-text,sentiment-sentiment,probability-probability, image=img_filename)

    ###############################################code for sentiment analysis

    if __name__=="__main__":
      init()
      app.run()

