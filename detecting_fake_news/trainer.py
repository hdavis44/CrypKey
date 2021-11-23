from detecting_fake_news.data import get_local_data, get_cloud_data
from detecting_fake_news.preprocessing import TextPreprocessor
from detecting_fake_news.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, LOCAL_TRAIN_DATA_PATH
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from termcolor import colored


def save_model_locally(model):
    '''save the model into a .joblib format'''
    joblib.dump(model, 'model.joblib')
    print(colored("model.joblib saved locally", "green"))


### TODO:

# add MLFlow functionality?

# create Trainer class with a pipeline attribute
#   class methods:
#       set_pipeline
#       run--sets and fits to pipeline
#       evaluate--evaluates on test data and returns accuracy



if __name__=='__main__':
    # get data
    print("getting cloud data")
    df = get_cloud_data(nrows=1000)
    df = df.dropna(subset=['text'])
    X = df['text']
    y = df['label']

    # preprocess data
    print("preprocessing data")
    preproc = TextPreprocessor()
    X_clean = preproc.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.25)

    # vectorize X_train
    print("vectorizing data")
    vectorizer = TfidfVectorizer(ngram_range=(2, 2)).fit(X_train)
    X_train_two_gram = vectorizer.transform(X_train)

    # fit vectorized X_train to MultinomialNB model
    print("fitting model")
    nb_model = MultinomialNB().fit(X_train_two_gram, y_train)

    # save model locally to model.joblib
    print("saving fitted model")
    save_model_locally(nb_model)

    # vectorize, predict, and score X_test
    print("vectorizing, predicting, and scoring on test data")
    X_test_two_gram = vectorizer.transform(X_test)
    y_pred = nb_model.predict(X_test_two_gram)
    score = accuracy_score(y_test, y_pred)

    # output accuracy
    print(colored(f"model accuracy: {score}", "green"))
