import re
import json
from tools.emoji import emoji

with open('tools/contractions.json', 'r') as f:
    contractions_dict = json.load(f)
contractions = contractions_dict['contractions']

with open('tools/stop_words.json', 'r') as f:
    stop_words_dict = json.load(f)

stopwords_list = []
for stop_words in stop_words_dict.values():
    stopwords_list.extend(stop_words)

class CustomPreprocess:
    def __init__(self):
        pass
        
    def process_tweet(self, tweet):
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
