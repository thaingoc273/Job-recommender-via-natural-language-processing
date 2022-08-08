import re
import numpy as np

import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import glob
import os

import pandas as pd
from deep_translator import GoogleTranslator #, DeeplTranslator, PonsTranslator
from langdetect import detect  

import time

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords


def clean_location(text):
    text = re.sub('(\d+|\+|Orte|Ort)', '', text)
    text = text.replace('Deutschland', 'Germany')
    text = text.replace('Heimarbeit', 'Homeoffice')
    text = text.split(',')[0]
    text = text.strip()    
    return text

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub('\([m|f|d|x|w]/[m|f|d|x|w]/[m|f|d|x|w]\)', '', text)
    text = re.sub(r"[()<>/]", ' ', text) # sub ()<>&/ to comma and space
    text = re.sub(r"&", 'and', text) # sub ()<>&/ to comma and space
    text = re.sub(r"[?!]", '. ', text) # sub ?! to dot and space
    text = re.sub("e\.g\.", '', text)
    text = re.sub("[\t\n\r\f\v]+", ". ", text)
    text = re.sub('\W+\.', '.', text) # remove the empty space before a dot
    text = re.sub('\W+\,', ',', text) # remove the empty space before a comma
    text = re.sub('[,\.]+\.+', '.', text) # sub multiple dots to one dot
    text = re.sub(' +',' ',text) # replace multiple whitespace by one whitespace
    
    text = [WordNetLemmatizer().lemmatize(token, "v") for token in text] #Lemmatization
    text = "".join(text)
    
    # Remove non english word
    # text = " ".join(w for w in nltk.wordpunct_tokenize(text) \
    #      if w.lower() in words or not w.isalpha())
    text = text.strip()
    return text

def change_string_of_list_to_string(lst_tring):
    return ', '.join(literal_eval(lst_tring))