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

app = Flask(__name__)

@app.route('/')
def home():
    return 'Book your appointments'
