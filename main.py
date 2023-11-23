import streamlit as st
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
@st.cache_data
def download_nltk_data():
    nltk.download('stopwords')
    
    stopwords.words('english')

download_nltk_data()

# Text processing function
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]

# Load dataset
@st.cache_data
def load_data():
    try:
        messages = pd.read_csv('SMSSpamCollection2.csv', sep='\t', names=['Label', 'Message'])
        messages['length'] = messages['Message'].apply(len)
        return messages
    except FileNotFoundError:
        return None

# Streamlit app
def main():
    st.title("SMS Spam Detection App")

    # Load data
    messages = load_data()
    if messages is None:
        st.error("Error: Dataset file not found.")
        return

    # Split dataset
    msg_train, msg_test, lab_train, lab_test = train_test_split(messages['Message'], messages['Label'], test_size=0.3)

    # Create and fit pipeline
    pipeline = Pipeline([
        ('bow', CountVectorizer(analyzer=text_process)),  
        ('tfidf', TfidfTransformer()),  
        ('classifier', MultinomialNB())  
    ])
    pipeline.fit(msg_train, lab_train)

    # User input
    user_input = st.text_input("Enter a message to check if it's spam or not:")

    if st.button("Predict"):
        if user_input:
            result = pipeline.predict([user_input])[0]
            st.write(f"The message is classified as: **{result}**")
        else:
            st.write("Please enter a message to classify.")

if __name__ == "__main__":
    main()
