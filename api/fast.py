# $DELETE_BEGIN

import os
import pandas as pd
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from detecting_fake_news.preprocessing import TextPreprocessor


app = FastAPI()


abspath = os.path.abspath("model.joblib")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)






@app.get("/predict")
def predictt(text):

    # text form strign to df
    text = pd.DataFrame(dict(text=[text]))

    #clean text, remove punctuation, etc
    clean_text = TextPreprocessor().transform(text)

    #load the model.joblib that is already train
    model = joblib.load(abspath)

    # predict the model with the new text
    prediction_local = model.predict(clean_text)

    # change the output type
    results = int(prediction_local[0])

    #api output
    return {'prediction' :results}
