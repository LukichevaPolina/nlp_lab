import pandas as pd
import logging as log
from string import punctuation
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

def drop_nan(data) -> pd.DataFrame:
    log.info("EDA: drop nan")
    return data.dropna()


def remove_punctuation(data) -> pd.DataFrame:
    log.info("EDA: remove punctuation")
    punctuations_str = punctuation
    punctuation_re = r'[{}]'.format(punctuations_str)
    replace_punct = lambda x: re.sub(punctuation_re, '', x)
    data["statement"] = data["statement"].map(replace_punct)

    return data

def remove_digits(data) -> pd.DataFrame:
    log.info("EDA: remove digits")
    digits_re = r'[0-9]+'
    replace_digits = lambda x: re.sub(digits_re, '', x)
    data["statement"] = data["statement"].map(replace_digits)

    return data

def remove_stop_words(data) -> pd.DataFrame:
    log.info("EDA: remove stop words")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    replace_stop_words = lambda x: ' '.join([word for word in x.lower().split(' ') if word  not in stop_words])
    data["statement"] = data["statement"].map(replace_stop_words)
    return data

def tokenize(data) -> pd.DataFrame:
    data["statement"] = data["statement"].map(lambda x: x.lower().split())
    return data

def stemming(data) -> pd.DataFrame:
    stemmer = PorterStemmer()
    data["statement"] = data["statement"].map(lambda x: [stemmer.stem(item) for item in x])
    return data

def lemmatization(data) -> pd.DataFrame:
    lemmer = WordNetLemmatizer()
    data["statement"] = data["statement"].map(lambda x: [lemmer.lemmatize(item) for item in x])
    return data