import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from wordcloud import WordCloud
import plotly.express as px

# visual
import matplotlib.pyplot as plt
import seaborn as sns

# filter out noise words and more clean up on word
from wordcloud import STOPWORDS,WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# gensim
import gensim
from gensim.models import CoherenceModel
from gensim import corpora

import string
import copy
import clean_data

topicData = ''


def process_data(topicData):
    topicData['clean_text'] = topicData['clean_text'].apply(lambda x: x.lower())
    topicData['clean_text'] = topicData['clean_text'].apply(lambda x: x.translate(str.maketrans(' ', ' ', string.punctuation)))
    
    sentence_list = [tweet for tweet in topicData['clean_text']]
    word_list = [sentence.split() for sentence in sentence_list]
    
    word_to_id = corpora.Dictionary(word_list)
    corpus_1= [word_to_id.doc2bow(tweet) for tweet in word_list]
    
    return word_list, word_to_id, corpus_1


def run():
    topicData = copy.deepcopy(clean_data.cleanTweet)
    st.write('# Topic Analysis')
    st.write('## Check Clean Tweet Head')
    st.write(topicData.head())

    topicData.dropna()
    process_data(topicData)
    word_list, id2word, corpus = process_data(topicData)

    lda_model = gensim.models.ldamodel.LdaModel(corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
  
    # # New Column 

    st.write('\nPerplexity: ', lda_model.log_perplexity(corpus))  

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=word_list, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    st.write('\n Ldamodel Coherence Score/Accuracy on Tweets: ', coherence_lda)
    
