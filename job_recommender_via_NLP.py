import selenium
from selenium import webdriver
from selenium.webdriver.chrome import service
import time
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
# import csv
import pandas as pd
from re import findall as re_findall
from datetime import date
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import spacy

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

def main():
    #(link, movie, rating, tag, rating_pivot, rating_agg) = load_data()
   
    st.title("Job Recommender via Natural Language Processing")
    

if __name__ == "__main__":
    main()