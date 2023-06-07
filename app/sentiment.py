import pytumblr
import re
import json
import tensorflow as tf
from keras.models import load_model
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from bs4 import BeautifulSoup
import pickle

#extract tweets and add them to a data frame

import pytumblr
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import detect

def extract_posts(tag, num_posts):
    consumer_key = ''
    consumer_secret = ''

    client = pytumblr.TumblrRestClient(consumer_key, consumer_secret)

    posts = client.tagged(tag, limit=num_posts)

    filtered_posts = []
    for post in posts:
        if 'type' not in post or post['type'] != 'text':
            continue

        if 'body' not in post:
            continue

        # Remove HTML tags from the body field
        soup = BeautifulSoup(post['body'], 'html.parser')
        cleaned_text = soup.get_text(separator=' ')

        # Detect the language of the cleaned text
        try:
            language = detect(cleaned_text)
        except:
            language = 'unknown'

        # Filter posts based on language (English)
        if language == 'en':
            filtered_posts.append({
                'ID': post['id'],
                'Username': post['blog_name'],
                'Hearts': post['note_count'],
                'full_text': cleaned_text
            })

    data = pd.DataFrame(filtered_posts)

    return data



#preprocessing the tweets using the same functions as in the ML model training

with open('/tools/contractions.json', 'r') as f:
    contractions_dict = json.load(f)
contractions = contractions_dict['contractions']

with open('/tools/stop_words.json', 'r') as f:
    stop_words_dict = json.load(f)

stopwords_list = []
for stop_words in stop_words_dict.values():
    stopwords_list.extend(stop_words)

def emoji(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :') , :O
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\)|:O)', 'positiveemoji', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' positiveemoji ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' positiveemoji ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-; , @-)
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;|@-\))', ' positiveemoji ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:, :-/ , :-|
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:|:-/|:-\|)', ' negativeemoji ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' negativeemoji ', tweet)
    return tweet

def process_tweet(tweet):
        tweet = tweet.lower()                                             # lowercase
        tweet = re.sub('@[^\s]+', '', tweet)                              # removes usernames
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', tweet)   # remove URLs
        tweet = re.sub(r"\d+", " ", str(tweet))                           # removes all digits
        tweet = re.sub('&quot;', " ", tweet)                              # Remove (&quot;) 
        tweet = emoji(tweet)                                              # replaces emojis
        tweet = re.sub(r"\b[a-zA-Z]\b", "", str(tweet))                   # Removes all single characters
        
        words = []
        for word in tweet.split():
            if word.lower() in contractions:
                replacement = contractions[word.lower()]
                words.extend(replacement.split('/'))
            else:
                words.append(word)
        
        tweet = ' '.join(words)
        
        tweet = re.sub(r"[^\w\s]", " ", str(tweet))                       # removes all punctuations
        tweet = re.sub(r'(.)\1+', r'\1\1', tweet)                         # convert more than 2 letter repetitions to 2 letter
        tweet = re.sub(r"\s+", " ", str(tweet))                           # replaces double spaces with single space    
        tweet = ' '.join(word for word in tweet.split() if word.lower() not in stopwords_list)  # remove stop words
        return tweet

#ready the text for the model

def process_tweet_data(data):
    lemmatizer = WordNetLemmatizer()
    word_tokenizer = Tokenizer()

    data['full_text'] = data['full_text'].apply(process_tweet)
    data['full_text'] = data['full_text'].apply(lambda x: word_tokenize(x))
    data['full_text'] = data['full_text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    word_tokenizer.fit_on_texts(data['full_text'])
    
    vocabulary = word_tokenizer.word_index
    
    data['full_text'] = data['full_text'].apply(lambda x: word_tokenizer.texts_to_sequences([x])[0])

    return data, vocabulary


def convert_sequence_to_text(sequence, tokenizer):
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    words = [index_to_word.get(idx, "") for idx in sequence]
    text = " ".join(words)
    return text

#load model and classify sentiments

def classify_sentiments(data,vocabulary):
    # Load the CountVectorizer
    vectorizer_file = 'count_vectorizer.pkl'
    with open(vectorizer_file, 'rb') as file:
        count_vectorizer = pickle.load(file)

    # Load the trained classifier
    classifier_file = 'naive_bayes_classifier.pkl'
    with open(classifier_file, 'rb') as file:
        model_naive = pickle.load(file)

    # Load the tokenizer information
    tokenizer_file = 'tokenizerNB.json'
    with open(tokenizer_file, 'r') as f:
        tokenizer_data = json.load(f)
        #vocabulary = tokenizer_data['vocabulary']
        ngram_range = tokenizer_data['ngram_range']
        
    sentiment_labels = {
        0: 'negative',
        1: 'positive'
        }
    
    word_tokenizer = Tokenizer()
    word_tokenizer.word_index = vocabulary
    oov_token = '<OOV>'  # or '<UNK>' or any other token you prefer
    word_tokenizer.oov_token = oov_token
    
    
    for index, sequence in enumerate(data['full_text']):
       text = convert_sequence_to_text(sequence, word_tokenizer)
       data.at[index, 'full_text'] = text
      
    sentiment_results = []    
    for post in data['full_text']:    
        post_str = ' '.join(str(token) for token in post)  # Convert list of integers to string
        feature_vector = count_vectorizer.transform([post_str])
        predicted_sentiment = model_naive.predict(feature_vector)
        predicted_label = sentiment_labels[1] if predicted_sentiment == 1 else sentiment_labels[0]
        sentiment_results.append(predicted_label)
    
    # Add the sentiment results as a new column in the DataFrame
    data['Sentiment'] = sentiment_results
    return data
    

def run(tag,num_tweets):
    data = extract_posts(tag, num_tweets)
    processed_data, vocabulary = process_tweet_data(data)
    classified_data = classify_sentiments(processed_data, vocabulary)
    return classified_data


