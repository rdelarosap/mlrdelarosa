#!/usr/bin/python
#coding: utf-8
import pickle
import flask
import gevent
import re as reg
import nltk as nltk
import string
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from flask import request, Flask, jsonify, abort, json
from flask_cors import CORS
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler # Para realizar normalización en escala (0-1)
from sklearn.preprocessing import normalize

app = flask.Flask(__name__)
cors = CORS(app, resources={r"/predicciones/*": {"origins": "*"}})

# Normalizar la data de entrada
normalizar = MinMaxScaler()

# Cargando el Modelo
#model = pickle.load(open('modelcocod.pkl','rb'))
def cargar_modelo(v_etapa):

    try:
        with open('model' + v_etapa.lower() + '.pkl', "rb") as f:
            return joblib.load(f)

    except (OSError, IOError) as e:
        return 'Error occurrido : ' + str(e)

# Cargando Diccionario
def cargar_diccionario(v_etapa):

    try:
        with open('dic' + v_etapa.lower() + '.dat', "rb") as f:
            return pickle.load(f)

    except (OSError, IOError) as e:
        return 'Error occurrido : ' + str(e) #dict()

# Token de las descripción
def tokenizar(texto):

    palabras = reg.split(r'\W+', str(texto))
    texto = [word.lower() for word in palabras]
    texto = nltk.Text(list(texto))
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in texto]
    words = [word for word in stripped if word.isalpha()]
    sin_articulos = set(stopwords.words('spanish'))
    words = [w for w in words if not w in sin_articulos]
    lemmatizer = WordNetLemmatizer()
    words_lemma = [lemmatizer.lemmatize(word, pos='v') for word in words]
    
    return words_lemma

# Transformación numérica
def trans_numerica(texto, _palabras, _categorias, _textos, tot_palabras):

    prob = 0

    for c in _categorias:
        # Probabilidad de la categoría
        prob_c = float(_categorias[c]) / float(_textos)
        palabras = tokenizar(texto)
        for p in palabras:
            # Probabilidad de la palabra
            if p in _palabras:
                prob_p =  (float(_palabras[p][c]) if float(_palabras[p][c]) != 0 else 0.5 )/ float(tot_palabras)
                # Probabilidad P(categoria|palabra)
                prob_cond = prob_p / prob_c
                # Probabilidad P(palabra|categoria)
                prob = (prob_cond * prob_p) / prob_c

    return prob

# Definiendo la ruta para una solicitud post
@app.route('/predicciones', methods=['POST'])
def index():

    try:

        # para trabajar con jwt
        #decoded = jwt.decode(data, 'secret', algorithm='HS256')
        data = request.get_json()
        feature = data['feature']

        if feature[1] == 1:
            v_eta = 5
            feature[1] = 'COCOD'
        elif feature[1] == 2:
            v_eta =  8
            feature[1] = 'PRSIS'
        elif feature[1] == 4:
            v_eta = 11
            feature[1] = 'ASEJE'
        elif feature[1] == 3:
            v_eta = 14
            feature[1] = 'APEJE'
        else:
            v_eta = 0
            feature[1] = ''
        
        model = cargar_modelo(feature[1])
        diccionario = cargar_diccionario(feature[1])

        v_categoria = diccionario.get('_categoria')
        v_textos = diccionario.get('n_palabra')
        v_tot_palabras = diccionario.get('r_entranmientos')


        del diccionario['_categoria']
        del diccionario['n_palabra']
        del diccionario['r_entranmientos']

        desc_num = trans_numerica(feature[0], diccionario, v_categoria, int(v_textos), int(v_tot_palabras))

        X = [desc_num, v_eta]

        response = {}
        #response['predictions'] = model.predict([8.11030008e-04, 1.00000000e+00]).tolist() # [8.11030008e-04 1.00000000e+00],[2.40064017e-03, 0.00000000e+00]
        response['predictions'] = X

    except ValueError as e:
        response = {'Try catch':"Error"}
        print(str(e))
        return flask.jsonify(response)

    return flask.jsonify(response)

if __name__ == '__main__': 
    app.run(debug=False)
    
