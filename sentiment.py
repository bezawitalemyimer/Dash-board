import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from wordcloud import WordCloud
import plotly.express as px

import copy
import clean_data

# Model traning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump, load # used for saving and loading sklearn objects
from scipy.sparse import save_npz, load_npz, csr_matrix # used for saving and loading sparse matrices
from sklearn.decomposition import NMF, LatentDirichletAllocation

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split



sentimentData = ""

def text_category(polarity):
    if (polarity > 0):
        return 'positive'
    elif (polarity < 0):
        return 'negative'
    else:
        return 'neutral' 

def train_and_show_scores(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    st.write(f'{title}\nTrain score: {round(train_score, 2)} ** - **  Validation score: {round(valid_score, 2)}\n')

def sentiment_analysis(sentimentData):
    # remove neutral 
    sentimentData = sentimentData[sentimentData['score'] != 'neutral']
    
    # add score map
    sentimentData['score_map'] = sentimentData['score'].apply(lambda x: 1 if x == 'positive' else 0)
    st.write("Adding Score map")
    st.write(sentimentData.head())

    # Text vecrotization
    st.write('## Text vectorization')
    # Input and output
    (X, y) = sentimentData['clean_text'], sentimentData['score_map']

    # Trigram Vecrorization 
    trigram_vectorization = CountVectorizer(ngram_range=(3,3))
    trigram_vectorization.fit(X.values)
    x_trigram = trigram_vectorization.transform(X.values)

    # Trigram TFIDF
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(x_trigram)
    x_tfidf = tfidf_transformer.transform(x_trigram)

    trigram_score = train_and_show_scores(x_trigram, y, title='TRIAGRAM')
    tfidf_score = train_and_show_scores(x_tfidf, y, title='TFIDF')

def run():
  sentimentData =  copy.deepcopy(clean_data.cleanTweet)
  st.write('# Sentiment Analysis')
  st.write('## Check Clean Tweet Head')
  st.write(sentimentData.head())
  
  # New Column 
  sentimentData['score'] = sentimentData['polarity'].apply(text_category)
  sentimentData['score'].value_counts()

  # bar chart
  score_count = sentimentData['score'].value_counts()
  st.write('## Polarty of datas')
  st.bar_chart(score_count)
  sentiment_analysis(sentimentData)    
