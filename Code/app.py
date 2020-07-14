## Phase 3: RESTful API
# Purpose: provide programmatic access to calcualte NER using our
# custom naive bayes classifier

# imports for RESTful API
from flask import Flask, jsonify, render_template, request
from flask_restful import Resource, Api
# imports to handle model
import pandas as pd
import joblib

# instantiate flask api
app = Flask(__name__)
api = Api(app)

# load naive bayes model
filename = "nb_ner.sav"
model = joblib.load(filename)

# load word vectors
word_vectors = joblib.load('word_vectors.joblib')

# load function to make the NER predictions
def make_NER_prediction(string, word_vectors):
    """Docstring: make a prediction on a target word. If the word is in our corpus, 
    the model provides the NER. Otherwise, the model provides'O'.
    
    This function uses the Naive Bayes model trained previously. 
    
    Inputs: string - input string to find NER
            word_vectors - word vectors set (saved as word_vectors.csv)
            
    Outputs: type string object with either NER prediction or 'O' for out of scope"""
    
    if string in word_vectors.Word.tolist():
        # if the word is in our corpus, grab the vector. there could be multiple occurances
        # so we are grabbing the mean of this vector
        x = word_vectors.loc[word_vectors.Word==string].mean().values
        # this vector is of the right dimensions that the model was fit on
        # now we can make a prediction
        pred = model.predict(x.reshape(1,-1))[0]
        return pred
    else:
        return "O"


class NER_API(Resource):
    """Create class to call NER function and make predictions"""
    def get(self):
        return {'About': 'This API perfoms NER on an input string'}
    
    def post(self):
        # POST request
        sentence = request.get_json()
        # preallocate empty list
        NER_labels = []
        # loop through all words in input sentence and make NER prediction
        for word in sentence.split(' '):
            NER_labels.append(make_NER_prediction(word, word_vectors))
        # jsonify response
        return {'Input Sentence': sentence,
                'NER Labels': NER_labels}, 201

api.add_resource(NER_API, '/')

# sample function call
# curl -H "Content-Type: application/json" -X POST -d '"Jack lives in London"' http://127.0.0.1:5000/

if __name__ == '__main__':
    app.run(debug=True)