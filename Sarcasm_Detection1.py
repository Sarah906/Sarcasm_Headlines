#### Maimona And Sarah Part ==========================
## Building prediction model
# In[1]:


# To store data
import pandas as pd

# To do linear algebra
import numpy as np

# To vectorize texts
from sklearn.feature_extraction.text import CountVectorizer

# To use new datatypes
from collections import Counter

# To stop words
import nltk

from nltk.corpus import stopwords
stop = stopwords.words('english')


import pickle

with open('data.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    logreg = pickle.load(f)
    
with open('counter.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    vectorizer = pickle.load(f)

#### Farah Part ==========================
## Building Flask Enviroment


from flask import Flask , request, jsonify
# import model

app = Flask(__name__)

#http://127.0.0.1:5000/predict?text=

@app.route('/predict',methods = ["GET", "POST"])
def hello_predict():
    text = request.args.get("text")
    test_text = vectorizer.transform([text.replace("'", '')])
    class_ = int(logreg.predict(test_text)[0])
    class_prob = logreg.predict_proba(test_text)
    class_prob_0 = float(class_prob[0][0])
    class_prob_1 = float(class_prob[0][1])
    
    result = {"is_sarcasm": class_,'probability_0': class_prob_0, 'probability_1': class_prob_1 }
    return jsonify(result)





if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0',port=80)

