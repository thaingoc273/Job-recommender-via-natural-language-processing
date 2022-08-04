from deep_translator import GoogleTranslator #, DeeplTranslator, PonsTranslator
import numpy as np
import pandas as pd

def google_translate_to_en(text, source_language):
    text_length = 4000
    if (len(text)<text_length):
        return GoogleTranslator(source=source_language, target='en').translate(text)
    else:            
        number =  len(text) // text_length
        mod = len(text) % text_length
        lst_text = [text[i*text_length: (i + 1)*text_length] for i in range(0, number)]
        lst_text.append(text[-mod:])
        text_translate = ''.join(GoogleTranslator(source=source_language, target='en').translate(item) for item in lst_text)
        return text_translate # GoogleTranslator(source='de', target='en').translate(text)

def google_translate_to_de(text, source_language):
    text_length = 4000
    if (len(text)<text_length):
        return GoogleTranslator(source=source_language, target='de').translate(text)
    else:            
        number =  len(text) // text_length
        mod = len(text) % text_length
        lst_text = [text[i*text_length: (i + 1)*text_length] for i in range(0, number)]
        lst_text.append(text[-mod:])
        text_translate = ''.join(GoogleTranslator(source=source_language, target='de').translate(item) for item in lst_text)
        return text_translate # GoogleTranslator(source='de', target='en').translate(text)

def job_description_translation(df):
    df['language'] = df['job_description'].apply(detect)
    df['job_description_en'] = df_linkedin_Berlin.apply(lambda x: x['job_description'] if x['language']=='en' else google_translate_to_en(x['job_description'], x['language']), axis = 1)
    df['job_description_de'] = df_linkedin_Berlin.apply(lambda x: x['job_description'] if x['language']=='de' else google_translate_to_de(x['job_description'], x['language']), axis = 1)
    return df