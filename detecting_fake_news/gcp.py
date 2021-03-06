import os
from google.cloud import storage
from termcolor import colored
# import joblib
# import tensorflow as tf
import pandas as pd

from detecting_fake_news.params import BUCKET_NAME

# BUCKET_NAME = 'wagon-data-745-fake-news-data'
# BUCKET_FOLDER = 'data'

# INFO >  file_name = 'model-xxx.joblib' or 'filename-of-data.csv'
# INFO >  model_directory = 'models' and 'data'


def storage_upload(cloud_directory, file_name, rm=False):
    client = storage.Client().bucket(BUCKET_NAME)
    storage_location = f"{cloud_directory}"
    blob = client.blob(storage_location)
    blob.upload_from_filename(file_name)
    print(
        colored(
            "=> {} uploaded to bucket {} inside {}".format(
                file_name, BUCKET_NAME, storage_location), "green"))
    if rm:
        os.remove(file_name)


def storage_download(cloud_directory, file_name):
    client = storage.Client().bucket(BUCKET_NAME)
    storage_location = f"{cloud_directory}"
    blob = client.blob(storage_location)
    blob.download_to_filename(file_name)
    print(
        colored(
            "=> {} downloaded from bucket {} inside {}".format(
                file_name, BUCKET_NAME, storage_location), "green"))


# ⭐️ Alternative model - simple, reading directly -
# source: https://stackoverflow.com/questions/51921142/how-to-load-a-model-saved-in-joblib-file-from-google-cloud-storage-bucket


#def read_bucket_model(model_file_name):
#    gcp_model_path = f"gs://{BUCKET_NAME}/models/{model_file_name}"
#
#    loaded_model = joblib.load(tf.io.gfile.GFile(gcp_model_path, 'rb'))
#    return loaded_model


def read_bucket_data(data_file_name, nrows):
    try:
        df = pd.read_csv(f"gs://{BUCKET_NAME}/data/{data_file_name}",
                         nrows=nrows)
        if nrows:
            print(colored(f"{nrows} rows loaded from gs://{BUCKET_NAME}/data/{data_file_name}","green"))
        else:
            print(colored(f"data loaded from gs://{BUCKET_NAME}/data/{data_file_name}","green"))
        return df
    except FileNotFoundError:
        print(colored(f"{data_file_name} not found in gs://{BUCKET_NAME}/data", "red"))
        return None
