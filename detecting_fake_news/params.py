### MLFLOW configuration - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = ""
EXPERIMENT_NAME = ""

### DATA & MODEL LOCATIONS  - - - - - - - - - - - - - - - - - - -

PATH_TO_LOCAL_MODEL = 'model.joblib'
AWS_BUCKET_TEST_PATH = ""

### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

PROJECT_ID = 'detecting-fake-news'

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-data-745-fake-news-data'
BUCKET_FOLDER = 'data'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/train.csv'
LOCAL_TRAIN_DATA_PATH = '<filepath><filename>'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = ''

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = ''

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

REGION = 'europe-west1'

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -
