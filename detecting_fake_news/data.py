from google.cloud import storage
import pandas as pd
from detecting_fake_news.gcp import read_bucket_data
from detecting_fake_news.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, LOCAL_TRAIN_DATA_PATH



def get_cloud_data(nrows=None, data_file_name='train.csv'):
    '''method to get training data from google cloud bucket'''
    if nrows:
        print(f"getting {nrows} rows of cloud data")
    else:
        print("getting cloud data")
    df = read_bucket_data(data_file_name, nrows)
    return df

def get_local_data(nrows=None):
    '''method to get training data from local machine'''
    if nrows:
        print(f"getting {nrows} rows of local data")
    else:
        print("getting local data")
    df = pd.read_csv(LOCAL_TRAIN_DATA_PATH, nrows=nrows)
    return df



if __name__=='__main__':
    df = get_cloud_data()
