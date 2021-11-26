from os import path, pipe
from detecting_fake_news.data import get_local_data, get_cloud_data, get_preprocessed_data
from detecting_fake_news.preprocessing import TextPreprocessor
from detecting_fake_news.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, LOCAL_TRAIN_DATA_PATH, PREPROCESSED_DATA_PATH
from detecting_fake_news.gcp import storage_upload
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
from termcolor import colored
import pandas as pd
import os




class Trainer(object):
    '''
    The Trainer class fits trains, evaluates, and saves an NLP model.
    The main method is Trainer.run.
    When instantiating an instance of the Trainer class, the user provides two
    positional arguments: X_col, and y_col.
    The user may also create an instance with data already set i.e. data=df.
    The user may also set the boolean keyword argument, preprocessed, to True if
    they are instantiating a Trainer instance using a preprocessed dataframe.
    '''
    def __init__(self, X_col, y_col, data=None, preprocessed=False):
        self.X_col = X_col
        self.y_col = y_col
        self.preprocessed = preprocessed

        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()
        self.pipe = None
        self.trained_model = None

        self.data = data

    def preprocess(self, new_line=True, punct=True, lower=True, accent=True,
                   numbers=True, stemm=False, lemm=True, stop_words=True):
        print("dropping rows of empty text")
        self.data = self.data.dropna(subset=[self.X_col])
        X = self.data[self.X_col]
        y = self.data[self.y_col]
        preproc = TextPreprocessor(new_line=new_line, punct=punct, lower=lower,
                                   accent=accent, numbers=numbers, stemm=stemm,
                                   lemm=lemm, stop_words=stop_words)
        print("preprocessing data with following parameters:")
        for k, v in vars(preproc).items():
            if v == True:
                print(f"{k}, ", end='')
        print('')
        X_clean = preproc.transform(X)
        self.data = pd.DataFrame({self.y_col: y, self.X_col: X_clean})
        self.preprocessed = True

    def set_pipeline(self):
        '''resets self.pipe and self.trained_model to None then sets self.pipe'''
        self.pipe = None
        self.trained_model = None
        pipe = Pipeline([('vectorizer', self.vectorizer),
                         ('model', self.model)])
        self.pipe = pipe

    def run(self):
        '''if self.data is not preprocessed, preprocesses according to params,
           sets self.data to the preprocessed data, and splits self.data into train/test;
           fits a pipeline to X_train, y_train; evaluates on X_test, y_test;
           prints an accuracy score'''
        if self.preprocessed == False:
            self.preprocess()
        self.data = self.data.dropna(subset=[self.X_col])
        X_clean = self.data[self.X_col]
        y = self.data[self.y_col]
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.25)
        print("setting pipeline")
        self.set_pipeline()
        print("vectorizing data and fitting model")
        self.trained_model = self.pipe.fit(X_train, y_train)
        print("evaluating on test data")
        self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        '''predicts y_pred based on X_test and scores accuracy on y_test'''
        if self.trained_model:
            y_pred = self.trained_model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print(colored(f"model accuracy: {score}", "green"))
        else:
            print("please train a model first using Trainer.run")

    def save_model_locally(self):
        '''save self.trained_model into a .joblib format'''
        if self.trained_model:
            joblib.dump(self.trained_model, 'model.joblib')
            print(colored("model.joblib saved locally", "green"))
        else:
            print("please train a model first using Trainer.run")

    def save_preprocessed(self, data_file_name):
        '''save a .csv file of preprocessed data in detecting_fake_news/data'''
        abspath = os.path.join(PREPROCESSED_DATA_PATH, data_file_name)
        if self.preprocessed:
            self.data.to_csv(abspath, columns=[self.y_col, self.X_col], index=False)
            print(colored(f"{data_file_name} saved in {PREPROCESSED_DATA_PATH}","green"))
        else:
            print(colored(f"preprocessed = {self.preprocessed} \nplease preprocess before saving","red"))

    def load_preprocessed(self, data_file_name, nrows=None):
        '''load a .csv file of preprocessed data from detecting_fake_news/data'''
        abspath = os.path.join(PREPROCESSED_DATA_PATH, data_file_name)
        try:
            self.data = pd.read_csv(abspath, nrows=nrows)
            if nrows:
                print(colored(f"{nrows} rows loaded from {PREPROCESSED_DATA_PATH}/{data_file_name}","green"))
            else:
                print(colored(f"data loaded from {PREPROCESSED_DATA_PATH}/{data_file_name}","green"))
            self.preprocessed = True
        except FileNotFoundError:
            print(colored(f"{data_file_name} not found in {PREPROCESSED_DATA_PATH}","red"))




if __name__=='__main__':
    df = get_cloud_data(nrows=3000)
    trainer = Trainer('text', 'label', data=df)
    trainer.run()
    #trainer.save_model_locally(trainer.trained_model)
    #storage_upload('models/MultinomialNB/model.joblib', 'model.joblib')
