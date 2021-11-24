from os import pipe
from detecting_fake_news.data import get_local_data, get_cloud_data
from detecting_fake_news.preprocessing import TextPreprocessor
from detecting_fake_news.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, LOCAL_TRAIN_DATA_PATH
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
from termcolor import colored





### TODO:

# add MLFlow functionality?

# create Trainer class with a pipeline attribute
#   class methods:
#       set_pipeline
#       run--sets and fits to pipeline
#       evaluate--evaluates on test data and returns accuracy


class Trainer(object):
    def __init__(self, X_col, y_col):
        self.X_col = X_col
        self.y_col = y_col
        self.pipe = None
        self.model = None

    def set_pipeline(self):
        '''resets self.pipe and self.model to None then sets self.pipe'''
        self.pipe = None
        self.model = None
        pipe = Pipeline([
            ('vectorizer', TfidfVectorizer(ngram_range=(2, 2))),
            ('nbmodel', MultinomialNB())])
        self.pipe = pipe

    def run(self, df):
        '''accepts a dataframe; preprocesses and splits data into train/test;
           fits a pipeline to X_train, y_train; evaluates on X_test, y_test;
           prints an accuracy score'''
        print("dropping rows of empty text")
        df = df.dropna(subset=[self.X_col])
        X = df[self.X_col]
        y = df[self.y_col]
        print("preprocessing data")
        preproc = TextPreprocessor()
        X_clean = preproc.transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_clean,
                                                            y,
                                                            test_size=0.25)
        print("setting pipeline")
        self.set_pipeline()
        print("vectorizing data and fitting model")
        self.model = self.pipe.fit(X_train, y_train)
        print("evaluating on test data")
        self.evaluate(X_test, y_test)

    def save_model_locally(self, model):
        '''save the model into a .joblib format'''
        joblib.dump(model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

    def evaluate(self, X_test, y_test):
        if self.model:
            X_vtest = self.model[0].transform(X_test)
            y_pred = self.model[1].predict(X_vtest)
            score = accuracy_score(y_test, y_pred)
            print(colored(f"model accuracy: {score}", "green"))
        else:
            print("please train a model first using Trainer.run")



if __name__=='__main__':
    df = get_cloud_data(nrows=3000)
    trainer = Trainer('text', 'label')
    trainer.run(df)
    trainer.save_model_locally(trainer.model)
