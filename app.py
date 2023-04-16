import random
import json
import pickle
import numpy as np
import nltk
import pyttsx3
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

from firebase import firebase

from flask import Flask, jsonify, request
firebase = firebase.FirebaseApplication('https://vocals-e4589-default-rtdb.asia-southeast1.firebasedatabase.app/', None)

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)  for word in sentence_words]
    return sentence_words

def bag_of_words(sentence,words):
    sentence_words= clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict_class(sentence,model,words,classes):
    bow = bag_of_words(sentence,words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda  x:x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    tag= intents_list[0]['intent']
    list_of_intents =intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
def firebase_response(entry):
    result = firebase.get('/prash/-NPmLDGT5Mvob4GFXAe_', None)
    res1 = result[entry]
    engine = pyttsx3.init()
    engine.say(res1)
    return res1
    
def chatbot_response(msg,model,intents,words,classes):
    if msg.lower() == "bye" or msg.lower()=="goodbye":
        ints = predict_class(msg, model,words,classes)
        res = "bye"
        return res
    
    else:
        ints = predict_class(msg, model,words,classes)
        res = get_response(ints, intents)
        if res == "name to be accessed":
            entry = 'name'
            return firebase_response(entry)
        elif res == "date to be accessed":
            entry = 'date'
            return firebase_response(entry)
        elif res == "time to be accessed":
            entry = 'time'
            return firebase_response(entry)
        elif res == "address to be accessed":
            entry = 'address'
            return firebase_response(entry)
        elif res == "number to be accessed":
            entry = 'person_no'
            return firebase_response(entry)
        elif res == "dept to be accessed":
            entry = 'department'
            return firebase_response(entry)
        elif res == "docname to be accessed":
            entry = 'doctor_name'
            return firebase_response(entry)
        elif res == "members to be accessed":
            entry = 'no_people'
            return firebase_response(entry)
        elif res == "issue to be accessed":
            entry = 'problem'
            return firebase_response(entry)
        elif res == "type to be accessed":
            entry = ''
            return firebase_response(entry)
        else:
            engine = pyttsx3.init()
            engine.say(res)
            engine.runAndWait()
            return res

app = Flask(__name__)

@app.route('/')
def home():
    return 'Book your appointments'

@app.route('/electrician/<name>', methods= ['GET']) 
def electrician(name):
    
    
    #dec_msg is the real question asked by the user
    dec_msg = name.replace("+", " ")
    
    intents = json.loads(open("electrician.json").read())
    words = pickle.load(open('electrician_words.pkl', 'rb'))
    classes = pickle.load(open('electrician_classes.pkl', 'rb'))
    model = load_model('electrician_chatbotmodel.h5')
    
    #get the response from the ML model & dec_msg as the argument
    response = chatbot_response(dec_msg,model,intents,words,classes)
    
    if response == "bye":
        # exit the application if the user says "bye"
        request.environ.get('werkzeug.server.shutdown')()
    
    
    return response
