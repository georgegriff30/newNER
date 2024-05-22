import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

st.title('NER model')
pipe = pipeline("token-classification", model="george6/roberta-finetuned-NER")
x = st.text_input('Enter a custom message:', '')
if x:
    list = []
    prediction = pipe(x)
    print(prediction)
    for i in prediction:
        list.append((i['word'], i['entity']))
    st.write(str(list))
