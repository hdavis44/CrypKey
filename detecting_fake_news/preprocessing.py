import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import unidecode
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
This function clean column(s) of text, row by row, according to
initiated parameters.

Clean is initiated by TextPreprocessor().transform(X)

Input (X) should be either:
- pandas.Series
- pandas.DataFrame

Cleaning process can handle handle both single and multiple column
DataFrames. Return format depends on input, se 'Returns below'

PARAMETERS
----------
new_line : bool, default True
    Remove all new-line characters in the text (\\n)

punct : bool, default True
    Remove all punctuation characters from text
    !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~

lower : bool, default True
    Change all characters to lower case

accent : bool, default True
    Replace all accent charcters with standard (non accented) characters

numbers : bool, default True
    Remove all numbers

stemm : bool, default False
    Perform Stemming on words 👉 cut to common root (sometimes not real word)

lemm : bool, default True
    Perform Lemming on words 👉 base word by meaning (laanguage correct)

stop_words : bool, default True
    Remove stopwords (hard set to English)
    -> run: set(stopwords.words('english')) - to review the words to be removed
    👍 useful for topic modelling, sentiment analysis
    👎 Useless for authorship attribution


RETURNS
-------
Return depends on input:
    - if pd.Series or pd.DataFrame with 1 column -> return pd.Series
    - if pd.DataFrame with 2 or more columns -> return pd.DataFrame
      (same size as input)
"""
    def __init__(self,
                 new_line=True,
                 punct=True,
                 lower=True,
                 accent=True,
                 numbers=True,
                 stemm=False,
                 lemm=True,
                 stop_words=True):

        self.new_line = new_line
        self.punct = punct
        self.lower = lower
        self.accent = accent
        self.numbers = numbers
        self.stemm = stemm
        self.lemm = lemm
        self.stop_words = stop_words

    def clean_text(self, text):
        if self.new_line:  # remove new-line: \n
            text = text.replace('\n', ' ')

        if self.punct:  # Remove Punctuation
            for punctuation in string.punctuation:
                text = text.replace(punctuation, ' ')

        if self.lower:  # Lower Case
            text = text.lower()

        if self.accent:  # Remove Accents
            text = unidecode.unidecode(text)

        # Tokenize --> (make word list) -> useful for following operation
        # token_text = word_tokenize(text)
        token_text = text.split()

        if self.numbers:  # Remove numbers
            token_text = [word for word in token_text if word.isalpha()]

        if self.stemm:  # Stemming 👉 cut to the common root (sometimes no at real word)
            stemmer = PorterStemmer()
            token_text = [stemmer.stem(word) for word in token_text]

        if self.lemm:  # Lemming 👉 base word by meaning (laanguage correct)
            lemmatizer = WordNetLemmatizer()
            token_text = [lemmatizer.lemmatize(word) for word in token_text]

        if self.stop_words:  # Remove Stop Words
            stop_words = set(stopwords.words('english'))
            token_text = [
                word for word in token_text if not word in stop_words
            ]

        return " ".join(token_text)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Check if input is DataFrame or Series
        if isinstance(X, pd.DataFrame):

            if X.shape[-1] > 1:  # DataFrame with 2 or more columns
                for col in X.columns:
                    X[col] = X[col].apply(self.clean_text)
                # will return a DataFrame
                return X

            else:  # DataFrame with 1 column
                X = X.iloc[:, 0]

        return X.apply(self.clean_text)
