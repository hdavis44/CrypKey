# $DELETE_BEGIN
import pytz

import pandas as pd
import joblib
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs

from detecting_fake_news.preprocessing import *


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


#TEST of api localhost

# @app.get("/predict")
# def predict(num1, num2):
#     X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
#     # fit final model
#     model = LogisticRegression()
#     model.fit(X, y)

#     # define one new instance
#     Xnew = pd.DataFrame(dict(num1=[float(num1)], num2=[float(num2)]))

#     # make a prediction
#     ynew = model.predict(Xnew)

#     ynew = int(ynew[0])
#     return {'prediction' :ynew}



@app.get("/predict")
def predictt(text):


    X = pd.DataFrame(dict(text=[text]))

    text = TextPreprocessor().transform(X)

    vectorizer = TfidfVectorizer(ngram_range=(2, 2)).fit(text)

    X_train_two_gram = vectorizer.transform(text)

    #text = text.apply(TextPreprocessor.clean_text())



    #model from the GCP
    model= joblib.load('model.joblib')


    # make prediction with the model of the GCP
    results = model.predict(X_train_two_gram)

    results=int(results[0])

    return {'prediction' :results}
