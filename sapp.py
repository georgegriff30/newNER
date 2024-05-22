import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

st.title('NER model')
pipe = pipeline("token-classification", model ="george6/NER", tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased"))
x = st.text_input('Enter a custom message:', 'Hello, Streamlit!')
if x:
    prediction = pipe(x)
    entities = [i['entity'] for i in prediction]
    st.write(str(entities))

