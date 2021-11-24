import joblib
import pandas as pd
import os
from detecting_fake_news.preprocessing import *

from google.cloud import storage
from tempfile import TemporaryFile



# run the predict from the local model.joblib
def predict_local(text):


    text = pd.DataFrame(dict(text=[text]))
    print('working 1', text)

    clean_text = TextPreprocessor().transform(text)
    print('working 2', clean_text)

    model = joblib.load('../model.joblib')
    print('working 3', model)

    prediction_local = model.predict(clean_text)
    print('working 4', prediction_local)

    results = int(prediction_local[0])
    print('working 5')

    return results



#still in process
# run the predict from the cloud model.joblib
storage_client = storage.Client()
bucket_name='<bucket name>'
model_bucket='model.joblib'

def predict_cloud():
    bucket = storage_client.get_bucket(bucket_name)
    #select bucket file
    blob = bucket.blob(model_bucket)
    #load that file from local file
    model = joblib.load(model_bucket)
