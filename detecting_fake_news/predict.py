import joblib
import pandas as pd
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from detecting_fake_news.preprocessing import TextPreprocessor
# from detecting_fake_news.params import BUCKET_NAME, PATH_TO_LOCAL_MODEL
from detecting_fake_news.gcp import storage_download
from detecting_fake_news.engineering import get_extended_eng_features_df

#function that make the api home page, it is used in fast.py
def home_page_api():
    return {
        '/test':
        'For testing if the API is working, add 2 parameter. num1=NUMBER, num2=NUMBER, retrun the sum of both numbers',
        '/predict_local':
        'For running the API using the model stored localy, add 1 parameter. text=TEXT, return 1=fake or 0=real',
        '/predict_cloud':
        'For running the API using the model from the cloud, add 1 parameter. text=TEXT, return 1=fake or 0=real',
        '/predict_proba_local':
        'For running the API using the model stored localy, add 1 parameter. text=TEXT, return proba of fake',
        '/predict_proba_cloud':
        'For running the API using the model from the cloud, add 1 parameter. text=TEXT, return proba of fake'
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


    ###### PROBA ######


#function that make the api prdict proba local page, it is used in fast.py
def predict_local_prob_api(text):
    abspath = os.path.abspath("model.joblib")
    # text form strign to df
    text = pd.DataFrame(dict(text=[text]))

    #clean text, remove punctuation, etc
    clean_text = TextPreprocessor().transform(text)

    #load the model.joblib that is already train
    model = joblib.load(abspath)

    # predict the model with the new text
    prediction_local = model.predict_proba(clean_text)

    # change the output type
    prova_dict= dict(enumerate(prediction_local.flatten(), 1))
    probability_fake= prova_dict[2]

    #probability of the text beign fake
    results = probability_fake

    #api output
    return {'prediction': results}


#function that make the api predict proba cloud page, it is used in fast.py
def predict_cloud_proba_api(text):
    abspath = os.path.abspath("cloud_model.joblib")

    # text form strign to df
    text = pd.DataFrame(dict(text=[text]))

    #clean text, remove punctuation, etc
    clean_text = TextPreprocessor().transform(text)

    #load the model.joblib that is already train
    storage_download('models/MultinomialNB/model.joblib', 'cloud_model.joblib')

    model = joblib.load(abspath)

    # predict the model with the new text
    prediction_local = model.predict_proba(clean_text)

    # change the output type
    prova_dict = dict(enumerate(prediction_local.flatten(), 1))
    probability_fake = prova_dict[2]

    #probability of the text beign fake
    results = probability_fake

    #api output
    return {'prediction': results}


## GET PREDICTIONS FROM ALL MODELS
def predict_all(text, source='cloud'):
    '''
    Get predictions from ALL models.
    - will clean text, ready for standards models
    - will create engineered features, for eng.feat. model
    - load models
    - make all predictions
    - make "total" weighted average prediction

    RETURN:
    Dictionary:
        - with all respective predictions + probabilities
        - total prediction + probability
    '''
    # text from string to pd.Series
    text = pd.Series(text)

    # Standard Preprocessing
    clean_text = TextPreprocessor().transform(text)

    # Get Extended Engineered Features
    eng_feat = get_extended_eng_features_df(raw_text=text, preproc_text=clean_text)

    # Load all the models
    models = {
        'multinomial': 'multinomial_model.joblib',
        'feat_eng': 'feat_eng_model.joblib',
        'LSTM_tokenizer': 'LSTM_tokenizer.joblib',
        'xgboost': 'xgboost_model.joblib'
    }
    if source == 'cloud':
        for model in models.values():
            storage_download(f'models_prod/{model}', model)

    multinomial_model = joblib.load(os.path.abspath(models['multinomial']))
    feat_eng_model = joblib.load(os.path.abspath(models['feat_eng']))
    xgboost_model = joblib.load(os.path.abspath(models['xgboost']))
    LSTM_model = load_model(os.path.abspath('LSTM_model'))
    LSTM_tokenizer = joblib.load(os.path.abspath(models['LSTM_tokenizer']))

    # Predict: MULTINOMIAL
    proba_multinomial = multinomial_model.predict_proba(clean_text)
    proba_multinomial = float(proba_multinomial[0][1])
    pred_multinomial = 1 if proba_multinomial >= 0.5 else 0

    # Predict: FEATURE ENGINEERING
    proba_feat_eng = feat_eng_model.predict_proba(eng_feat)
    proba_feat_eng = float(proba_feat_eng[0][1])
    pred_feat_eng = 1 if proba_feat_eng >= 0.5 else 0

    # Predict: XGBOOST
    proba_xgboost = xgboost_model.predict_proba(clean_text)
    proba_xgboost = float(proba_xgboost[0][1])
    pred_xgboost = 1 if proba_xgboost >= 0.5 else 0

    # Get LSTM features
    X_token = LSTM_tokenizer.texts_to_sequences(clean_text)
    X_pad = pad_sequences(X_token, dtype='float32', padding='post', maxlen=500)

    # Predict: LSTM
    proba_LSTM = LSTM_model.predict(X_pad)[0][1]
    pred_LSTM = 1 if proba_LSTM >= 0.5 else 0

    # Make weighted mean prediction
    wm_proba = (95 * proba_multinomial +
                90 * proba_xgboost +
                80 * proba_feat_eng +
                98 * proba_LSTM) / (95 + 90 + 80 + 98)
    wm_pred = 1 if wm_proba >= 0.5 else 0

    return {
        'multinomial_pred': pred_multinomial,
        'multinomial_proba': proba_multinomial,
        'xgboost_pred': pred_xgboost,
        'xgboost_proba': proba_xgboost,
        'feat_eng_pred': pred_feat_eng,
        'feat_eng_proba': proba_feat_eng,
        'LSTM_proba': proba_LSTM,
        'LSTM_pred': pred_LSTM,
        'mean_pred': wm_pred,
        'mean_proba': wm_proba,
    }
