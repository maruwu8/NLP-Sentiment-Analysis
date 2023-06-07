import streamlit as st
import sentiment

st.set_page_config(
    page_title="Tumblr sentiment analysis",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
   
)

st.title("Tumblr Sentiment Analysis 🐱")

tag = st.sidebar.text_input("Enter the Hashtag or any topic")

number_of_tweets = st.slider("How many blogs do you want to collect for {}".format(tag), min_value=100, max_value=300)


if st.button("Analyze Sentiment"):
    data1 = sentiment.extract_posts(tag, number_of_tweets)
    st.write(data1)
    data = sentiment.run(tag, number_of_tweets)
    st.write(data)
    
