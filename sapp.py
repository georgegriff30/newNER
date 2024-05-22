import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

st.title('NER model')
pipe = pipeline("token-classification", model="george6/NER", tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"))
x = st.text_input('Enter a custom message:', '')
if x:
    list = []
    prediction = pipe(x)
    print(prediction)
    for i in prediction:
        list.append((i['word'], i['entity_group']))
    st.write(str(list))
