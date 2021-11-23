# $DELETE_BEGIN
import pytz

import pandas as pd
import joblib

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs


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
def index(num1,num2):
    # generate 2d classification dataset
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
    # fit final model
    model = LogisticRegression()
    model.fit(X, y)

    # define one new instance
    Xnew = [[num1, num2]]
    # make a prediction
    ynew = model.predict(Xnew)

    return ("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
