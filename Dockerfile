FROM python:3.8.12-buster

COPY api /api
COPY detecting_fake_news /detecting_fake_news
COPY requirements.txt /requirements.txt
# Hard copy the models...
COPY LSTM_model.h5 /LSTM_model.h5
COPY LSTM_tokenizer.joblib /LSTM_tokenizer.joblib
COPY feat_eng_model.joblib /feat_eng_model.joblib
COPY multinomial_model.joblib /multinomial_model.joblib
COPY xgboost_model.joblib /xgboost_model.joblib
# Hard copy the nltk_data
COPY ./nltk_data /usr/local/nltk_data

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# install Spacy data
RUN python -m spacy download en_core_web_sm

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
