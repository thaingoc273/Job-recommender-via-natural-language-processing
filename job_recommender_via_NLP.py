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
from langdetect import detect   ## use for language detection

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

from sklearn.neighbors import NearestNeighbors
from scipy.stats.stats import pearsonr   

import sys
sys.path.insert(0, 'ultility') # Add path ultility for add functions
import language
from language import google_translate_to_en, google_translate_to_de, job_description_translation

import fitz

def main(): 
       
        #st.sidebar.markdown("## Choose language")
        language = st.sidebar.selectbox("Language", ["English", "German", "Both"])
        
        if (language=='English'):
            df = df_job_en.copy()            
        elif (language=='German'):
            df = df_job_de.copy()
        else:
            df = df_job.copy()
        
        #st.sidebar.markdown("## Choose city")
        lst_city = list(df['location'].unique())
        lst_city.insert(0, 'All cities')
        city = st.sidebar.selectbox("City", lst_city)
        if city != 'All cities':
            df = df.loc[df['location'].str.contains(city)]       
        
        level = st.sidebar.selectbox('Type of level', ['Junior', 'Senior', 'Both'])
        if (level == 'Senior'):
            df = df.loc[df['position'].str.contains('(Senior|Lead|Experienced|Principal|Head|Director)', flags=re.IGNORECASE, regex=True)==True]
        elif (level == 'Junior'):
            df = df.loc[df['position'].str.contains('(Senior|Lead|Experienced|Principal|Head|Director)', flags=re.IGNORECASE, regex=True)==False]
        
            
#         st.sidebar.markdown("## Choose type of algorithms")
#         algorithm = st.sidebar.selectbox("", ["Cosine", "KNN", "Pearson"])
        
#         if (algorithm=='Cosine'):
#             top_job =  cosin_similarity(df, cv_skill_text, number)
#         elif (algorithm=='KNN'):
#             top_job = KNN_similartity(df, cv_skill_text, number)
#         else:
#             top_job =  person_corr_similarity(df, cv_skill_text, number)
      
        top_job = cosin_similarity(df, cv_skill_text, number)
        
        # st.sidebar.text_area('Your skill', cv_skill_text)
        
        top_job = top_job.reset_index()
        top_job.rename(columns={'position':'Position', 'company_name':'Company', 'location':'Location', 'skill_extraction':'Skill','link':'Link'}, inplace = True)
        #top_job_link_format = top_job[['position', 'company_name', 'location', 'skill_extraction' ,'link']].style.format({'link': make_clickable})
        top_job_link_format = top_job[['Position', 'Company', 'Location', 'Link']].style.format({'Link': make_clickable})
        top_job_link_format = top_job_link_format.to_html(escape=False)
        st.write(top_job_link_format, unsafe_allow_html=True)
        

def make_clickable(val):
    return f'<a target="_blank" href="{val}">Link to apply</a>'
            
@st.cache
def load_file(upload_file, typ):
    # document = fitz.open(stream=upload_file.read(), filetype="pdf")
    # page_count = document.page_count
    # CV_text = ''
    # for i in range(page_count):
    #     CV_text+= document.load_page(i).get_text()
    # st.text_area(CV_text)
    # return CV_text


    if (typ != 'application/pdf') & (typ != 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
        st.error('File is not in correct format. Please upload again')
        return None
    
    elif (typ=='application/pdf'):        
        document = fitz.open(stream=upload_file.read(), filetype="pdf")
        page_count = document.page_count
        CV_text = ''
        for i in range(page_count):
            CV_text+= document.load_page(i).get_text()
        return CV_text
        
        # CV_pdf = pdfplumber.open(upload_file)
        # num_page = CV_pdf.pages
        # CV_text = ''
        # for i in range(len(num_page)):
        #     pageObj = CV_pdf.pages[i]
        #     CV_text += pageObj.extract_text()
        # return CV_text
        
    elif (typ == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
        CV_text = docx2txt.process(upload_file)
        return CV_text


@st.cache    
def skill_extraction_one(text):
    text = text.replace('’', "'")
    
    df_skill_full_match = pd.json_normalize(skill_extractor.annotate(text)['results']['full_matches'])
    try:
        df_skill_ngram = pd.json_normalize(skill_extractor.annotate(text)['results']['ngram_scored'])
        df_skill = pd.concat([df_skill_ngram, df_skill_full_match])
    except:
        df_skill = df_skill_full_match
    # df_skill_full_match = pd.json_normalize(skill_extractor.annotate(text)['results']['full_matches'])
    # df_skill = pd.concat([df_skill_ngram, df_skill_full_match])
    df_skill = df_skill.merge(df_skill_data, how = 'left', left_on='skill_id', right_on='skill_id')
    
    return df_skill['skill_name'].unique().tolist()


def skill_extractor_model():
    nlp = spacy.load("en_core_web_md")
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    return skill_extractor

def cosin_similarity(df, cv_skill_text, number):
    # Calculate tfidf
    df = df.reset_index()
    tfidf_skill = tfidf_model.transform([cv_skill_text])
    lst_job_description_en = df['skill_extraction'].tolist()
    tfidf_job_description = tfidf_model.transform(lst_job_description_en)
    
    # Calculate similarity
    cosin_sim = cosine_similarity(tfidf_skill, tfidf_job_description)
    
    df['cosin_similarity'] = np.array(cosin_sim).ravel()
    
    
    return df.sort_values(by='cosin_similarity', ascending=False).head(number)

def KNN_similartity(df, cv_skill_text, number):
    df = df.reset_index()
    tfidf_skill = tfidf_model.transform([cv_skill_text])
    lst_job_description_en = df['skill_extraction'].tolist()
    tfidf_job_description = tfidf_model.transform(lst_job_description_en)
    
    nearest_neighbor = NearestNeighbors(n_neighbors=number)
    nearest_neighbor.fit(tfidf_job_description)
    result_KNN = nearest_neighbor.kneighbors(tfidf_skill)
    lst_index = np.array(result_KNN[1]).ravel().tolist()
    
    return df.loc[lst_index, :]

def person_corr_similarity(df, cv_skill_text, number):
    df = df.reset_index()
    tfidf_skill = tfidf_model.transform([cv_skill_text])
    lst_job_description_en = df['skill_extraction'].tolist()
    tfidf_job_description = tfidf_model.transform(lst_job_description_en)
    
    tfidf_skill_dense = tfidf_skill.todense().tolist()[0]
    
    lst_pearson_corr = []
    for job_description in tfidf_job_description:
        job_description_dense = job_description.todense().tolist()[0]
        lst_pearson_corr.append(pearsonr(tfidf_skill_dense,job_description_dense)[0])
    df['pearson_corr'] = lst_pearson_corr
    
    return df.sort_values(by='pearson_corr', ascending=False).head(number)

@st.cache
def load_dataframe():
    df_job = pd.read_csv('data/skill_extraction_Skiller_08.08_final_web.csv')
    df_job_en = df_job.loc[df_job['language']=='en'].copy()
    df_job_de = df_job.loc[df_job['language']=='de'].copy()
    df_skill_data = pd.read_csv('data/skill_name.csv') # read all skills
    return df_job, df_job_en, df_job_de, df_skill_data

    
if __name__ == "__main__":
    number = 20
    st.set_page_config(layout="wide")
    
    nlp = spacy.load("en_core_web_sm")
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    
    # model = pickle.load(open('model/skill_extractor_sm.pkl','rb'))
    
    tfidf_model = pickle.load(open('model/tfidf_model_0808.pkl','rb'))
    
    
    df_job, df_job_en, df_job_de, df_skill_data = load_dataframe()
    
    # df_job = pd.read_csv('data/skill_extraction_Skiller_03.08_final_web_v1.csv')
    # df_job_en = df_job.loc[df_job['language']=='en'].copy()
    # df_job_de = df_job.loc[df_job['language']=='de'].copy()
    # df_skill_data = pd.read_csv('data/skill_name.csv') # read all skills
    
        
    col1, col2 = st.columns(2)
    with col1:
        st.image('picture/logo_WBS.png')
    with col2:
        st.title('For a Better World')    
    
    check = st.sidebar.checkbox("STATISTICS")
    
    upload_file = st.sidebar.file_uploader('Please upload CV in .pdf or .docx file', type=["pdf","docx"])
    
    if (check == False) & (upload_file is None):        
        st.image('picture/images.jpg')    
    elif (check == True):
        st.title('1. Languages in job description')
        col3, col4 = st.columns(2)
        with col3:
            st.image('picture/language_ds.png')
        with col4:
            st.image('picture/language_da.png')
        
        st.title('2. Location and Job title')
        col3, col4 = st.columns(2)
        with col3:
            st.image('picture/location.png')
        with col4:
            st.image('picture/job_title.png')
        
        st.title('3. Technical skills')
        col3, col4 = st.columns(2)
        with col3:
            st.image('picture/ds_tech_skill.png')
        with col4:
            st.image('picture/da_tech_skill.png')
        
        st.title('4. Some soft skills')
        col3, col4 = st.columns(2)
        with col3:
            st.image('picture/ds_soft_skill.png')
        with col4:
            st.image('picture/da_soft_skill.png')
            
    elif (upload_file is not None):
            cv_text = load_file(upload_file, upload_file.type)            
            lang_detect = detect(cv_text)
            if (lang_detect!='en'):
                cv_text = google_translate_to_en(cv_text, lang_detect)            

            cv_skill = skill_extraction_one(cv_text)
            cv_skill_text = ', '.join(cv_skill)

            main()