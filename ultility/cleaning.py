import re
import numpy as np


def clean_location(text):
    text = re.sub('(\d+|\+|Orte|Ort)', '', text)
    text = text.replace('Deutschland', 'Germany')
    text = text.replace('Heimarbeit', 'Homeoffice')
    text = text.split(',')[0]
    text = text.strip()    
    return text