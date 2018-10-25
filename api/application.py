# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier
from flask import Flask, jsonify, request, url_for, render_template
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import RussianStemmer
import pymorphy2
import re
import nltk
import json
from sklearn.externals import joblib
import boto3
import _pickle
import requests


BUCKET_NAME = 'textsentimentmodels'
MODEL_FILE_NAME = 'Decision_Tree_Full.pkl'
TFIDF_FILE_NAME = 'TFIDF_Vectorizer_Full.pkl'

ACCESS_KEY = 'ACCESS_KEY'
SECRET_KEY = 'SECRET_KEY'

# EB looks for an 'application' callable by default.
application = Flask(__name__)

S3 = boto3.client('s3', region_name = 'eu-west-3', aws_access_key_id = ACCESS_KEY, aws_secret_access_key = SECRET_KEY)



def prepare_text(text):
    stopword = set([u'и', u'в', u'во', u'не', u'что', u'он', u'на', u'я', u'с', u'со', u'как', u'а', u'то', u'все', u'она', u'так', u'его', u'но', u'да', u'ты', u'к', u'у', u'же', u'вы', u'за', u'бы', u'по', u'только', u'ее', u'мне', u'было', u'вот', u'от', u'меня', u'еще', u'нет', u'о', u'из', u'ему', u'теперь', u'когда', u'даже', u'ну', u'вдруг', u'ли', u'если', u'уже', u'или', u'ни', u'быть', u'был', u'него', u'до', u'вас', u'нибудь', u'опять', u'уж', u'вам', u'ведь', u'там', u'потом', u'себя', u'ничего', u'ей', u'может', u'они', u'тут', u'где', u'есть', u'надо', u'ней', u'для', u'мы', u'тебя', u'их', u'чем', u'была', u'сам', u'чтоб', u'без', u'будто', u'чего', u'раз', u'тоже', u'себе', u'под', u'будет', u'ж', u'тогда', u'кто', u'этот', u'того', u'потому', u'этого', u'какой', u'совсем', u'ним', u'здесь', u'этом', u'один', u'почти', u'мой', u'тем', u'чтобы', u'нее', u'сейчас', u'были', u'куда', u'зачем', u'всех', u'никогда', u'можно', u'при', u'наконец', u'два', u'об', u'другой', u'хоть', u'после', u'над', u'больше', u'тот', u'через', u'эти', u'нас', u'про', u'всего', u'них', u'какая', u'много', u'разве', u'три', u'эту', u'моя', u'впрочем', u'хорошо', u'свою', u'этой', u'перед', u'иногда', u'лучше', u'чуть', u'том', u'нельзя', u'такой', u'им', u'более', u'всегда', u'конечно', u'всю', u'между'])
    text = text.lower()
    text = re.sub(r"[\Wa-zA-Z\d^\s_]", " ", text, flags=re.U)
    text = re.sub(r"й", u"и", text, flags=re.U)
    text = re.sub(r"ё", u"е", text, flags=re.U)
    text = " ".join(w for w in nltk.wordpunct_tokenize(text) if not(w.lower() in stopword))
    r = RussianStemmer()
    text = r.stem(text)
    morph = pymorphy2.MorphAnalyzer()
    text = " ".join([morph.parse(y)[0].normal_form for y in text.split(" ")])
    
    
    return text


@application.route('/', methods=['GET', 'POST'])
def index():
    errors = []
    results = ""
    if request.method == "POST":
        try:
            text = request.form['text']
            r = requests.post("https://7pyovt43zb.execute-api.eu-west-3.amazonaws.com/dev/sentiment", data = {'text' : text})
            results = r.content.decode('Utf-8')
            
            if (json.loads(results)['score']) > 0.5:
                return render_template('index.html', errors = errors, results = ("positive", "{0:.2f}".format(json.loads(results)['score'] * 100)))
            else:
                return render_template('index.html', errors = errors, results = ("negative", "{0:.2f}".format(json.loads(results)['score'] * 100)))
        except:
            errors.append(
                "Unable to get Text. Please make sure it's valid and try again."
            )
    
    return render_template('index.html', errors = errors, results = results)
    

@application.route('/sentiment', methods=['GET', 'POST'])
def analyze():   
    #print(url_for('analyze', /sentiment))
    if request.method == 'GET':
        response = requests.post("https://7pyovt43zb.execute-api.eu-west-3.amazonaws.com/dev/", data = {'text' : request.form['text']})
        return response.content.decode('UTF-8')
    else:
        text = request.form['text']
        text = prepare_text(text)
        
        model = load_model(MODEL_FILE_NAME)
        tfidf = load_model(TFIDF_FILE_NAME)
        
        tr = np.array(tfidf.transform([text]).todense(), dtype = np.float16)
        
        res = model.predict_proba(tr)
        
        return jsonify({'score': res[0][1]})


def load_model(key):    
    # Load model from S3 bucket
    response = S3.get_object(Bucket = BUCKET_NAME, Key = key)
    # Load pickle model
    model_str = response['Body'].read()     
    model = _pickle.loads(model_str)     
    
    return model

# run the app.
if __name__ == "__main__":
    application.run(debug=True)
