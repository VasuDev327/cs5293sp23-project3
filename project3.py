import os
import PyPDF2

from pypdf import PdfReader


import re 

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import pickle
from pathlib import Path


import argparse

from pypdf import PdfReader
import os
import re
import pandas as pd

def extract_text_function(filename):
    if filename.endswith(".pdf"):
        with open(filename, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PdfReader(file)
            # Get the total number of pages in the PDF
            num_pages = len(pdf_reader.pages)
            text = ""
            # Loop through each page in the PDF
            for page_num in range(num_pages):
                # Get the text from the current page
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        return text

"""
Created on Mon Aug 01 01:11:02 2016

@author: DIP
"""

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}

import nltk
import spacy
import unicodedata
# from contractions import CONTRACTION_MAP
import re
from nltk.corpus import wordnet
import collections
#from textblob import Word
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup

custom_stopwords = ['city', 'smart', 'cities', 'states', 'page','US','transportation', 'vehicles', 'vehicle']
tokenizer = ToktokTokenizer()
nlp = spacy.load("en_core_web_md")
# nlp_vec = spacy.load('en_vectors_web_lg', parse=True, tag=True, entity=True)
stopword_list = set(nltk.corpus.stopwords.words('english'))
stopword_list.update(custom_stopwords)


def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    if bool(soup.find()):
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    else:
        stripped_text = text
    return stripped_text

def simple_porter_stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
            
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                    flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]|\[|\]' if not remove_digits else r'[^a-zA-Z\s]|\[|\]'
    text = re.sub(pattern, '', text)
    return text


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


def normalize_corpus(corpus):
    text = strip_html_tags(corpus)
    text = text.translate(text.maketrans("\n\t\r", "   "))
    text = remove_accented_chars(text)
    text = expand_contractions(text)
    text = lemmatize_text(text)
    text = simple_porter_stemming(text)
    text = remove_special_characters(text) 
    text = re.sub(' +', ' ', text)
    text = remove_stopwords(text)        
    return text

def smart_city_slicker(argument):
    filename = argument.document
    text = []
    for f in filename:
        text.append(extract_text_function(f))
    dataframe = pd.DataFrame()
    city_name = []
    state_name = []
    for f in filename:
        text_between = f[:f.find(".pdf")]
        state = text_between.split()[0]
        city = text_between.split()[1]
        state_name.append(state)
        city_name.append(city)
    raw_text = []
    for t in text:
        raw_text.append(t)
    dataframe['State Name'] = state_name
    dataframe['City Name'] = city_name
    dataframe['Raw Text'] = raw_text
    # print(dataframe)
    dataframe['Cleared Text'] = dataframe['Raw Text'].apply(normalize_corpus)
    # print("done with cleaning")
    # loading the pickle file which has been saved while training the data in jupyter code
    # cluster id
    with open('model.pkl', 'rb') as file:
        loaded_model, loaded_vectorizer = pickle.load(file)
    X = loaded_vectorizer.transform(dataframe['Cleared Text'])
    cluster = loaded_model.predict(X)
    dataframe['Cluster ID'] = cluster
    # print("done with model.pkl")
    
    # topic
    with open('LDA model.pkl', 'rb') as f:
        lda_loaded, vectorizer_loaded_l = pickle.load(f)
    # print("done with LDA")

    # Transform text data using the loaded vectorizer
    X = vectorizer_loaded_l.transform(dataframe['Cleared Text'])

    # Get the topic distribution for each document
    doc_topic_dist = lda_loaded.transform(X)

    # Get the top two topics for each document based on topic distribution
    top_two_topics = np.argsort(-doc_topic_dist, axis=1)[:, :2]

    # Create topic IDs for the top two topics and add them to the dataframe
    topic_ids = ['T' + str(topic_idx) for topic_idx in top_two_topics.flatten()]
    topic_ids = np.array(topic_ids)

    dataframe['Topic ID'] = np.split(topic_ids, len(dataframe))
    
    # Print the Output
    row = dataframe.loc[0]

    city = row['City Name']
    state = row['State Name']
    clusterid = row['Cluster ID']
    output = f"[{city}, {state}] clusterid: {clusterid}]"
    print(output)
    dataframe['City_State'] = dataframe['City Name'].str.cat(dataframe['State Name'], sep=', ')
    if os.path.isfile('smartcity_predict.tsv'):
        dataframe.to_csv('smartcity_predict.tsv', sep='\t', index=False, mode='a', 
                header=False, 
                columns=['City_State','Raw Text', 'Cleared Text', 'Cluster ID', 'Topic ID'], escapechar='\\')
    else:
        dataframe.to_csv('smartcity_predict.tsv', sep='\t', index=False, 
                columns=['City_State','Raw Text', 'Cleared Text', 'Cluster ID', 'Topic ID'], escapechar='\\')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='The Smart City Slicker')
    arg_parser.add_argument('--document', required=True, action="append", help='new pdf file name')
    arg_parser.add_argument('--summarize', action='append', help='summarize')
    arg_parser.add_argument('--keywords', action='append', help='keywords')
    args = arg_parser.parse_args()

    if args.document:
        smart_city_slicker(args)
