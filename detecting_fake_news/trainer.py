from os import pipe
from detecting_fake_news.data import get_local_data, get_cloud_data
from detecting_fake_news.preprocessing import TextPreprocessor
from detecting_fake_news.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, LOCAL_TRAIN_DATA_PATH
from detecting_fake_news.gcp import storage_upload
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import joblib
from termcolor import colored



### TODO:

# add MLFlow functionality?



class Trainer(object):
    '''
    The Trainer class fits trains, evaluates, and saves an NLP model.
    The main method is Trainer.run, which takes a dataframe as an argument.
    When instantiating an instance of the Trainer class, you provide two
    arguments, X_col and y_col, which correspond to the column names of your
    feature and label respectively.
    '''
    def __init__(self, X_col, y_col):
        self.X_col = X_col
        self.y_col = y_col
        self.pipe = None
        self.model = None

    def set_pipeline(self):
        '''resets self.pipe and self.model to None then sets self.pipe'''
        self.pipe = None
        self.model = None
        pipe = Pipeline([('vectorizer', TfidfVectorizer(ngram_range=(2, 2))),
                         ('nbmodel', MultinomialNB())])
        self.pipe = pipe

    def run(self, df, preprocessed=False,
            new_line=True, punct=True, lower=True, accent=True,
            numbers=True, stemm=False, lemm=True, stop_words=True):
        '''accepts a dataframe; preprocesses and splits data into train/test;
           fits a pipeline to X_train, y_train; evaluates on X_test, y_test;
           prints an accuracy score'''
        print("dropping rows of empty text")
        df = df.dropna(subset=[self.X_col])
        X = df[[self.X_col]]
        y = df[self.y_col]
        if preprocessed == False:
            preproc = TextPreprocessor(new_line=new_line, punct=punct,
                                       lower=lower, accent=accent,
                                       numbers=numbers, stemm=stemm,
                                       lemm=lemm, stop_words=stop_words)
            print("preprocessing data with following parameters:")
            for k, v in vars(preproc).items():
                if v == True:
                    print(f"{k}, ", end='')
            print('')
            X_clean = preproc.transform(X)
        else:
            X_clean = X
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y, test_size=0.25)
        print("setting pipeline")
        self.set_pipeline()
        print("vectorizing data and fitting model")
        self.model = self.pipe.fit(X_train, y_train)
        print("evaluating on test data")
        self.evaluate(X_test, y_test)

    def evaluate(self, X_test, y_test):
        '''predicts y_pred based on X_test and scores accuracy on y_test'''
        if self.model:
            y_pred = self.model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print(colored(f"model accuracy: {score}", "green"))
        else:
            print("please train a model first using Trainer.run")

    def save_model_locally(self, model):
        '''save the model into a .joblib format'''
        joblib.dump(model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))



if __name__=='__main__':
    df = get_cloud_data(nrows=3000)
    trainer = Trainer('text', 'label')
    trainer.run(df)
    trainer.save_model_locally(trainer.model)
    storage_upload('models/MultinomialNB/model.joblib', 'model.joblib')
