# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import streamlit as st
# Load the Logistic regression model and Tfidf-vectorizer object from disk
filename = 'Model/LG_model.pickle'
classifier = pickle.load(open(filename, 'rb'))
tf = pickle.load(open('Model/tfidf-transformer.pickle','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        txt_len = len(data[0].split())
        vect = tf.transform(data).toarray()
        my_prediction = classifier.predict(vect)

        if txt_len>0:
            return render_template('result1.html', prediction=my_prediction)
        else:
            return render_template('result1.html', prediction='')
if __name__ == '__main__':
	app.run(debug=True)