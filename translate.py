from transformers import MarianMTModel, MarianTokenizer

# Load saved model
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

# User input
text = input("Enter English Sentence: ")

translation = translate_text(text)

print("Hindi Translation:", translation)