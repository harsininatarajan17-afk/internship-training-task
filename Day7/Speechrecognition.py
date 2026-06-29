import streamlit as st
import speech_recognition as sr
import tempfile

st.title("Speech To Text")

uploaded_file = st.file_uploader(
    "Upload WAV Audio",
    type=["wav"]
)

if uploaded_file:

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    temp.write(uploaded_file.read())

    r = sr.Recognizer()

    with sr.AudioFile(temp.name) as source:

        audio = r.record(source)

        try:
            text = r.recognize_google(audio)

            st.success(text)

        except:
            st.error("Could not recognize speech")