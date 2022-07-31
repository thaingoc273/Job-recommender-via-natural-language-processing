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

import streamlit as st

import PyPDF2
from io import StringIO
import textract
import docx2txt
import pdfplumber

def main():
    #(link, movie, rating, tag, rating_pivot, rating_agg) = load_data()
   
    st.title("Job Recommender via Natural Language Processing")
    uploaded_file = st.file_uploader('Please upload CV in .pdf or .doc, .docx file', type=["pdf","docx", "doc"])
    if uploaded_file is not None:
        if (uploaded_file.type=='application/pdf'):
            CV_pdf = pdfplumber.open(uploaded_file)
            pageObj = CV_pdf.pages[0]
            CV_text = pageObj.extract_text()
            st.text_area(CV_text)
        elif (uploaded_file.type=='application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
            # CV_text = textract.process(uploaded_file)
            # CV_text = CV_text.decode("utf-8") 
            # st.text_area(CV_text)            
            CV_text = docx2txt.process(uploaded_file)
            st.text_area(CV_text)

if __name__ == "__main__":
    main()