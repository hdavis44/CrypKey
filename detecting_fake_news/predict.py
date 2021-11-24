import joblib
import pandas as pd
import os
from detecting_fake_news.preprocessing import *





# run the predict from the local model.joblib
def predict_local(text):

    clean_text = TextPreprocessor().clean_text(text)

    model = joblib.load('DETECTING_FAKE_NEWS/model.joblib')

    prediction_local= model.predict(clean_text)

    results = int(prediction_local[0])

    return results

# run the predict from the cloud model.joblib
def predict_cloud():
    pass
