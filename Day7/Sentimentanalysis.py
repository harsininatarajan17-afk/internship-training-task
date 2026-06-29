import streamlit as st
from textblob import TextBlob

st.title("Sentiment Analysis")

text = st.text_area("Enter Review")

if st.button("Analyze"):

    analysis = TextBlob(text)

    score = analysis.sentiment.polarity

    if score > 0:
        st.success("Positive")
    elif score < 0:
        st.error("Negative")
    else:
        st.warning("Neutral")

    st.write("Polarity:", score)