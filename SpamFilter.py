# coding: utf-8

# Importing required libraries
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
if not nltk.data.find('corpora/stopwords'):
    nltk.download('stopwords')

# Function to process text
def text_process(mess):
    """
    Process the input text by removing punctuation and stopwords.
    :param mess: input message string
    :return: list of words after processing
    """
    # Remove punctuation
    nopunc = [char for char in mess if char not in string.punctuation]
    # Join characters to form the string again
    nopunc = ''.join(nopunc)
    # Remove stopwords and return the list of words
    return [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]

# Load and prepare the dataset
try:
    messages = pd.read_csv('SMSSpamCollection2.csv', sep='\t', names=['Label', 'Message'])
except FileNotFoundError:
    print("Dataset file not found.")
    exit()

# Add a new column for message length
messages['length'] = messages['Message'].apply(len)

# Splitting the dataset into training and test sets
msg_train, msg_test, lab_train, lab_test = train_test_split(messages['Message'], messages['Label'], test_size=0.3)

# Creating a pipeline for preprocessing, transforming, and classifier
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # Convert messages to Bow format
    ('tfidf', TfidfTransformer()),  # Apply TF-IDF
    ('classifier', MultinomialNB())  # Classifier
])

# Fitting the pipeline to training data
pipeline.fit(msg_train, lab_train)

# User interaction for classification
x = input("Enter your message to find out whether it is spam or not: ")
prediction = pipeline.predict([x])
print(f"The message is classified as: {prediction[0]}")
