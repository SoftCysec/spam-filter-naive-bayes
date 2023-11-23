
# coding: utf-8

# In[6]:

import nltk
import string
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
nltk.download('stopwords')
#stopwords corpus is the list of words used for removing the conjuctures, prepositions etc from a sentence 


# In[7]:


#importing the dataset for training. 
messages = pd.read_csv('SMSSpamCollection2',sep='\t',names=['Label','Message'])


# In[8]:

messages['length']= messages['Message'].apply(len)
#applying the length function to get the length of all the messages. 


# In[9]:

def text_process(mess):
    #remove all the punctuation
    #remove all the stop words
    #return the list of important words
    
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunch = ''.join(nopunc)
    return [word for word in nopunch.split() if word.lower() not in stopwords.words("english")]
#this method will return each word devoid of the stopwords and punctuations in  the string.


# In[10]:




# In[11]:

bow_transf = CountVectorizer(analyzer=text_process).fit(messages['Message'])
#this count vectorizer creates a document -term matrix 


# In[12]:

messages_bow = bow_transf.transform(messages['Message'])
#now we will create a sparse matrix to get the term vs document frequency
# we will be using this to find the The inverse document frequency and Tfid transformer ratio.


# In[13]:




# In[14]:

tfid_transformer = TfidfTransformer().fit(messages_bow)
#fitting the transformer with the sparse matrix " messages_bow"


# In[15]:

messages_tfidf = tfid_transformer.transform(messages_bow)
#fitting the transformer 


# In[16]:


#importing the multinomialNB classifier which is usually used for textual analysis and prediction


# In[17]:




# In[18]:

msg_train,msg_test,lab_train,lab_test = train_test_split(messages['Message'],messages['Label'],test_size=0.3)


# In[19]:




# In[20]:

pip = Pipeline([('bow',CountVectorizer(analyzer=text_process)),('tfidf',TfidfTransformer()),('classifier',MultinomialNB())]

#training the model using the pipeline feature of the Sklearn library. 
)


# In[21]:

pip.fit(msg_train,lab_train) #fitting the data.


# In[28]:

x = input("Enter you message to find out wether it is spam or not! ")


# In[29]:

y = []
y.append(x)


# In[30]:

output = pip.predict(y)


# In[31]:

print(output)


# In[ ]:



