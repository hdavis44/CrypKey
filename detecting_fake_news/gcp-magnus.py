import os
from google.cloud import storage
from termcolor import colored
import joblib
import tensorflow as tf

from detecting_fake_news.params import BUCKET_NAME

# BUCKET_NAME = 'wagon-data-745-fake-news-data'
# BUCKET_FOLDER = 'data'

# INFO >  file_name = 'model-xxx.joblib' or 'filename-of-data.csv'
# INFO >  model_directory = 'models' and 'data'


def storage_upload(model_directory, file_name, rm=False):
    client = storage.Client().bucket(BUCKET_NAME)
    storage_location = f"{model_directory}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(file_name)
    print(colored("=> model.joblib uploaded to bucket {} inside {}".format(
        BUCKET_NAME, storage_location),"green"))
    if rm:
        os.remove(file_name)


def storage_download(model_directory, file_name):
    client = storage.Client().bucket(BUCKET_NAME)
    storage_location = f"{model_directory}"
    blob = client.blob(storage_location)
    blob.download_to_filename(file_name)
    print(colored("=> model.joblib downloaded from bucket {} inside {}".format(
        BUCKET_NAME, storage_location), "green"))


# ⭐️ Alternative model - simple, reading directly -
# source: https://stackoverflow.com/questions/51921142/how-to-load-a-model-saved-in-joblib-file-from-google-cloud-storage-bucket
def read_bucket_model(model_file_name):
    gcp_model_path = f"gs://{BUCKET_NAME}/models/{model_file_name}"

    loaded_model = joblib.load(tf.io.gfile.GFile(gcp_model_path, 'rb'))
    return loaded_model
