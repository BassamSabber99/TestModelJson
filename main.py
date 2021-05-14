from flask import Flask,request,jsonify
from flask_cors import CORS

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model



app = Flask(__name__)
CORS(app)
@app.route('/validateString', methods=['GET', 'POST'])
def validateString():

   text = request.get_data()

   train = pd.read_csv('train.csv')
   classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
   train_sentences = train['comment_text']
   max_features = 22000

   tokenizers = Tokenizer(num_words = max_features)
   tokenizers.fit_on_texts(list(train_sentences))
   tokenized = tokenizers.texts_to_sequences(str(text))
   case = pad_sequences(tokenized, maxlen=200) 
   
   model = load_model('LSTM.h5')
   predict = model.predict(case)
   predict = np.where(predict>=0.1)
   coord = list(zip(predict[0],predict[1]))
   output = []
   for x,y in coord:
     output.append(classes[y])

   output = list(set(output))
   return  jsonify(output)



if __name__ == '__main__':
   app.run()
