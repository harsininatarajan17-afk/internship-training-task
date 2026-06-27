import streamlit as st
import numpy as np
import pandas as pd
import re
import nltk
import joblib
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Input
nltk.download("punkt", quiet=True)

st.header("73. RNN")
X = np.array([[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]]])
y = np.array([4,5,6])
model = Sequential([Input(shape=(3,1)),SimpleRNN(10),Dense(1)])
model.compile(loss="mse", optimizer="adam")
model.fit(X, y, epochs=200, verbose=0)
seq = st.text_input(
    "Enter 3 numbers separated by comma",
    "4,5,6",
    key="rnn_input"
)
if st.button("Predict", key="rnn_btn"):
    values = [float(x) for x in seq.split(",")]
    arr = np.array(values).reshape(1,3,1)
    pred = model.predict(arr)
    st.success(f"Next Value: {pred[0][0]:.2f}")
st.divider()

st.header("74. LSTM")
X = np.array([[[10],[20],[30]],[[20],[30],[40]],[[30],[40],[50]]])
y = np.array([40,50,60])
model = Sequential([
    Input(shape=(3,1)),
    LSTM(20),
    Dense(1)
])
model.compile(loss="mse", optimizer="adam")
model.fit(X, y, epochs=200, verbose=0)
data = st.text_input(
    "Enter last 3 prices",
    "40,50,60",
    key="lstm_input"
)
if st.button("Predict Price", key="lstm_btn"):
    values = [float(x) for x in data.split(",")]
    arr = np.array(values).reshape(1,3,1)
    pred = model.predict(arr)
    st.success(f"Predicted Price: {pred[0][0]:.2f}")
st.divider()

st.header("75. Time Series")
df = pd.DataFrame({"Month":[1,2,3,4,5],"Sales":[100,120,140,160,180]})
X = df[["Month"]]
y = df["Sales"]
model = LinearRegression()
model.fit(X,y)
month = st.number_input("Enter Future Month",min_value=1,value=6,key="month_input")
if st.button("Predict Sales", key="sales_btn"):
    pred = model.predict(pd.DataFrame({"Month":[month]}))
    st.success(f"Predicted Sales: {pred[0]:.2f}")
st.divider()

st.header("76. NLP Basics")
text = st.text_area("Enter Text",key="nlp_text")
if st.button("Analyze", key="nlp_btn"):
    words = len(text.split())
    chars = len(text)
    st.write("Words:", words)
    st.write("Characters:", chars)
    st.write("Uppercase:")
    st.write(text.upper())
    st.write("Lowercase:")
    st.write(text.lower())
st.divider()

st.header("77. Text Preprocessing")
text = st.text_area("Enter Text",key="preprocess_text")
if st.button("Process", key="process_btn"):
    cleaned = text.lower()
    cleaned = re.sub(r'[^a-z ]', '', cleaned)
    st.success(cleaned)
st.divider()

st.header("79. Tokenization")
text = st.text_area(
    "Enter Text",
    key="token_text"
)
if st.button("Tokenize", key="token_btn"):
    tokens = text.split()
    st.write(tokens)

st.header("80. Word Embeddings")
sentences = [["machine","learning"],["deep","learning"],["artificial","intelligence"]]
model = Word2Vec(sentences,vector_size=10,min_count=1)
word = st.selectbox("Select Word",model.wv.index_to_key,key="word_select")
if st.button("Show Vector", key="vector_btn"):
    st.write(model.wv[word])
st.divider()

st.header("83. Evaluation")
actual = st.text_input("Actual","1,0,1,1,0",key="actual_input")
predicted = st.text_input("Predicted","1,0,1,0,0",key="predicted_input")
if st.button("Evaluate", key="eval_btn"):
    y_true = [int(x) for x in actual.split(",")]
    y_pred = [int(x) for x in predicted.split(",")]
    st.write("Accuracy:", accuracy_score(y_true,y_pred))
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_true,y_pred))

st.header("84. Save / Load Model")
X = pd.DataFrame({"x":[1,2,3,4]})
y = [2,4,6,8]
model = LinearRegression()
model.fit(X,y)
joblib.dump(model, "model.pkl")
loaded = joblib.load("model.pkl")
value = st.number_input("Enter Value",value=5.0,key="model_input")
if st.button("Predict Model", key="model_btn"):
    pred = loaded.predict(pd.DataFrame({"x":[value]}))
    st.success(f"Prediction: {pred[0]:.2f}")