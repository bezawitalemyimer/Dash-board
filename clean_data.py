''' 
This file will clean the data for 
semantic and topic analysis purposes.
'''
import pandas as pd
from add_data import db_execute_fetch
import re

def clean_text(text):
    hash_tag_removed = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', text)
    removed_links = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', hash_tag_removed, flags=re.MULTILINE)
    result = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', removed_links)
    return result

def loadData():
    query = "select * from TweetInformation"
    df = db_execute_fetch(query, dbName="tweets", rdf=True)
    return df

def info():
  number_of_empty_texts = df['clean_text'].isnull().sum()
  shape = df.shape
  return (number_of_empty_texts, shape)

df = loadData()

cleanTweet = pd.DataFrame()
cleanTweet['clean_text'] = df['clean_text'].apply(clean_text)
cleanTweet['polarity'] = df['polarity']
