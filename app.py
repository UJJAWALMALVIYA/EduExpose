import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Load model
model_path = "./model"

tokenizer = MarianTokenizer.from_pretrained(model_path)
model = MarianMTModel.from_pretrained(model_path)

# Translation function
def translate_text(text):

    inputs = tokenizer(text, return_tensors="pt")

    translated = model.generate(**inputs)

    output = tokenizer.decode(
        translated[0],
        skip_special_tokens=True
    )

    return output

# Streamlit UI
st.title("English to Hindi Translator")

user_input = st.text_input("Enter English Text")

if st.button("Translate"):

    result = translate_text(user_input)

    st.success(result)