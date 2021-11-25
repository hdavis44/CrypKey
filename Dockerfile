FROM python:3.8.12-buster

COPY api /api
COPY detecting_fake_news /detecting_fake_news
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
# Hard copy the nltk_data
COPY ./nltk_data /usr/local/nltk_data

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
