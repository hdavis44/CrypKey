import joblib
import pandas as pd
import os

from detecting_fake_news.preprocessing import TextPreprocessor
# from detecting_fake_news.params import BUCKET_NAME, PATH_TO_LOCAL_MODEL
from detecting_fake_news.gcp import storage_download


#function that make the api home page, it is used in fast.py
def home_page_api():
    return {
        '/test':
        'For testing if the API is working, add 2 parameter. num1=NUMBER, num2=NUMBER',
        '/predict_local':
        'For running the API using the model stored localy, add 1 parameter. text=TEXT',
        '/predict_cloud':
        'For running the API using the model from the cloud, add 1 parameter. text=TEXT'
    }


#function that make the api test page, it is used in fast.py
def tester_api(num1, num2):
    return {'result':(int(num1) + int(num2))}


#function that make the api prdict local page, it is used in fast.py
def predict_local_api(text):
    abspath = os.path.abspath("model.joblib")
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
    return {'prediction': results}



#function that make the api predict cloud page, it is used in fast.py
def predict_cloud_api(text):
    abspath = os.path.abspath("cloud_model.joblib")

    # text form strign to df
    text = pd.DataFrame(dict(text=[text]))

    #clean text, remove punctuation, etc
    clean_text = TextPreprocessor().transform(text)

    #load the model.joblib that is already train

    storage_download('models/MultinomialNB/model.joblib', 'cloud_model.joblib')

    model = joblib.load(abspath)

    # predict the model with the new text
    prediction_local = model.predict(clean_text)

    # change the output type
    results = int(prediction_local[0])

    #api output
    return {'prediction': results}
