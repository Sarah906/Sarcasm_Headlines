#### Maimona And Sara Part ==========================
## Building prediction model


# coding: utf-8

# # Sarcasm Detection

# ## Import libraries

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

# ## Load the data

# In[2]:


data = pd.read_json('Sarcasm_Headlines_Dataset.json',lines=True)


# ## Vectorize headlines

# In[3]:


# Create vectorizer
countVectorizer = CountVectorizer(stop_words=stop)

# Vectorize text
vectorizedText = countVectorizer.fit_transform(data['headline'].str.replace("'", '').values)


# In[7]:


vec = pd.DataFrame(vectorizedText.toarray(),columns=countVectorizer.get_feature_names())


# In[8]:


NLP = vec


# ## Train split

# In[9]:


X = NLP
y = data.is_sarcastic


# In[11]:


# In[12]:


# ## LogisticRegression

# In[ ]:


# import the class
from sklearn.linear_model import LogisticRegression

# In[ ]:


# instantiate the model
logreg = LogisticRegression()
# fit the model to the data
logreg.fit(X,y)

#### Farah Part ==========================
## Building Flask Enviroment


from flask import Flask , request, jsonify
# import model

app = Flask(__name__)

#http://127.0.0.1:5000/predict?text=hi esraa

@app.route('/predict',methods = ["GET", "POST"])
def hello_predict():
    text = request.args.get("text")
    test_text = countVectorizer.transform([text.replace("'", '')])
    class_ = int(logreg.predict(test_text)[0])
    class_prob = logreg.predict_proba(test_text)
    class_prob_0 = float(class_prob[0][0])
    class_prob_1 = float(class_prob[0][1])
    
    result = {"is_sarcasm": class_,'probability_0': class_prob_0, 'probability_1': class_prob_1 }
    return jsonify(result)

