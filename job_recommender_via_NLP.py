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
import pickle

from spacy.matcher import PhraseMatcher

# load default skills data base
from skillNer.general_params import SKILL_DB

# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def main():   
    st.title("Job Recommender via Natural Language Processing")
    
    # with st.form('Upload CV'):
    upload_file = st.sidebar.file_uploader('Please upload CV in .pdf or .docx file', type=["pdf","docx"])
    if upload_file is not None:
        cv_text = load_file(upload_file, upload_file.type)

        cv_skill = skill_extraction_one(cv_text)
        cv_skill_text = ', '.join(cv_skill)
        # st.sidebar.title('Your skills')
        st.sidebar.text_area('Your Skill', cv_skill_text)

        top_job =  cosin_similarity(df_job, cv_skill_text, number)
        st.write(top_job)
            #st.text_area(top_job)

def load_file(upload_file, typ):
    if (typ != 'application/pdf') & (typ != 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
        st.error('File is not in correct format. Please upload again')
        return None
    
    elif (typ=='application/pdf'):
        CV_pdf = pdfplumber.open(upload_file)
        num_page = CV_pdf.pages
        CV_text = ''
        for i in range(len(num_page)):
            pageObj = CV_pdf.pages[i]
            CV_text += pageObj.extract_text()
        return CV_text
    elif (typ == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
        CV_text = docx2txt.process(upload_file)
        return CV_text
    
def skill_extraction_one(text):
    df_skill_ngram = pd.json_normalize(skill_extractor.annotate(text)['results']['ngram_scored'])
    df_skill_full_match = pd.json_normalize(skill_extractor.annotate(text)['results']['full_matches'])
    df_skill = pd.concat([df_skill_ngram, df_skill_full_match])
    
    return df_skill['doc_node_value'].unique().tolist()


def skill_extractor_model():
    nlp = spacy.load("en_core_web_md")
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    return skill_extractor

def cosin_similarity(df, cv_skill_text, number):
    # Calculate tfidf
    tfidf_skill = tfidf_model.transform([cv_skill_text])
    lst_job_description_en = df['skill_extraction'].tolist()
    tfidf_job_description = tfidf_model.transform(lst_job_description_en)
    
    # Calculate similarity
    cosin_sim = cosine_similarity(tfidf_skill, tfidf_job_description)
    
    df['cosin_similarity'] = np.array(cosin_sim).ravel()
    
    
    return df.sort_values(by='cosin_similarity', ascending=False).head(number)

if __name__ == "__main__":
    number = 15
    nlp = spacy.load("en_core_web_sm")
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    #model = pickle.load(open('model/skill_extractor_sm.pkl','rb'))
    tfidf_model = pickle.load(open('model/tfidf_model.pkl','rb'))
    df_job = pd.read_csv('data/skill_extraction_Skiller_03.08_final_web.csv')
    main()