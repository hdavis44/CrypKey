FROM python:3.8.12-buster

COPY api /api
COPY detecting_fake_news /detecting_fake_news
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# need to run-download nltk data
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader stopwords

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
