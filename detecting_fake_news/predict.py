import joblib
import pandas as pd

from detecting_fake_news.preprocessing import TextPreprocessor
from detecting_fake_news.params import BUCKET_NAME, PATH_TO_LOCAL_MODEL

from google.cloud import storage
import tensorflow as tf


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


#still in process
def read_bucket_model(model_file_name):
    gcp_model_path = f"gs://{BUCKET_NAME}/models/{model_file_name}"

    loaded_model = joblib.load(tf.io.gfile.GFile(gcp_model_path, 'rb'))
    return loaded_model


#still in process
def predict_cloud(text):
    text = pd.DataFrame(dict(text=[text]))
    print('working 1', text)

    clean_text = TextPreprocessor().transform(text)
    print('working 2', clean_text)

    model = joblib.load('../model.joblib')
    print('working 3', model)

    prediction_local = read_bucket_model(PATH_TO_LOCAL_MODEL).predict(clean_text)
    print('working 4', prediction_local)

    results = int(prediction_local[0])
    print('working 5')

    return results
