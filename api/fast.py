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
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


#TEST of api localhost

@app.get("/predict")
def predict(num1, num2):
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
    # fit final model
    model = LogisticRegression()
    model.fit(X, y)

    # define one new instance
    Xnew = pd.DataFrame(dict(num1=[int(num1)], num2=[int(num2)]))

    # make a prediction
    ynew = model.predict(Xnew)

    return dict(prediction=ynew[0])



# @app.get("/predict")
# def predict(text):

#     text=TextPreprocessor.clean_text(text)

#     X = pd.DataFrame(
#         dict(text))

#     #model from the GCP
#     model= joblib.load('model.joblib')

#     # make prediction with the model of the GCP
#     results = model.predict(X)
