from google.cloud import storage
import os
import pandas as pd
from termcolor import colored
from detecting_fake_news.gcp import read_bucket_data
from detecting_fake_news.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, LOCAL_TRAIN_DATA_PATH, PREPROCESSED_DATA_PATH



def get_cloud_data(data_file_name='train.csv', nrows=None):
    '''method to get training data from google cloud bucket'''
    df = read_bucket_data(data_file_name, nrows)
    return df


def get_local_data(data_file_name='train.csv', nrows=None):
    '''method to get training data from local machine'''
    abspath = os.path.join(LOCAL_TRAIN_DATA_PATH, data_file_name)
    try:
        df = pd.read_csv(abspath, nrows=nrows)
        if nrows:
            print(colored(f"{nrows} rows loaded from {LOCAL_TRAIN_DATA_PATH}/{data_file_name}","green"))
        else:
            print(colored(f"data loaded from {LOCAL_TRAIN_DATA_PATH}/{data_file_name}","green"))
        return df
    except FileNotFoundError:
        print(colored(f"{data_file_name} not found in {LOCAL_TRAIN_DATA_PATH}","red"))
        return None


def get_preprocessed_data(data_file_name='preproc_train.csv', nrows=None):
    '''method for getting preprocessed data from local machine'''
    abspath = os.path.join(PREPROCESSED_DATA_PATH, data_file_name)
    try:
        df = pd.read_csv(abspath, nrows=nrows)
        if nrows:
            print(colored(f"{nrows} rows loaded from {PREPROCESSED_DATA_PATH}/{data_file_name}","green"))
        else:
            print(colored(f"data loaded from {PREPROCESSED_DATA_PATH}/{data_file_name}","green"))
        return df
    except FileNotFoundError:
        print(colored(f"{data_file_name} not found in {PREPROCESSED_DATA_PATH}","red"))
        return None



if __name__=='__main__':
    df = get_local_data()
    print(df.head())
