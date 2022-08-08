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

def all_version_library(): # return all version of library for streamlit app
    print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
    
def skill_summary(df):
    lst_skill = df.loc[df['skill_extraction'].isna()==False]['skill_extraction'].tolist()
    lst_skill = [literal_eval(item) for item in lst_skill]
    lst_skill_unpack = [item for sub_list in lst_skill for item in sub_list]
    df_skill = pd.DataFrame(data=lst_skill_unpack, columns=['skill'])
    return df_skill.groupby('skill')['skill'].count().sort_values(ascending=False)