## ** necessary Libraries and Modules**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""## **Loading the dataset**"""

data = 'save-the-csv-file-and-give-path-of-the-dataset-here'
# dataset link : https://www.kaggle.com/datasets/charunisa/chatgpt-sentiment-analysis?select=file.csv

df = pd.read_csv(data)

df.drop(df[df.labels == 'neutral'].index, inplace = True)
df.head(100)

"""## **Displaying the info about the Dataset**"""

df.info()

"""## **Checking the missing values**"""

df.isnull().sum()

df = df.dropna(subset=['tweets'])

df.info()

"""## **Checking label distribution**"""

df['labels'].value_counts()

"""## **Visualizing the label distribution**"""

sns.countplot(x='labels', data=df)

"""## **Cleaning the Dataset**"""

import re

def clean_text(text):
    text = re.sub(r'Ã[\x80-\xBF]+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text.lower()

"""## **Removing URLs**"""

import re

def remove_url(text):

    # regular expression to find URLs (http/https/ftp)
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(url_pattern, '', text)

df['tweets'] = df['tweets'].apply(lambda x: remove_url(str(x)))

print("URLs removed successfully!")
df.head()

# Applying the clean_text function to Review column
df['tweets'] = df['tweets'].apply(lambda x: clean_text(str(x)))

# To view full text
pd.set_option('display.max_colwidth', None)

df.head(5)

"""## **Tokenization**"""

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

df['Tokens'] = df['tweets'].apply(lambda x: tokenize_text(str(x)))

df.head()

pip install wordcloud

"""## **Plotting the WordCloud**"""

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re

img = 'wordcloud-image-file-png/jpeg/jpg'

comment_words = ''
stopwords = set(STOPWORDS)

for val in df.Tokens:

   # Converting each token to lowercase
    tokens = [str(token).lower() for token in val]

    # Cleaning up any unwanted characters (removing extra single quotes, brackets, etc.)
    # This removes non-alphanumeric characters
    tokens = [re.sub(r"[^\w\s]", '', token) for token in tokens]

    # Joining all tokens into a string and appending to comment_words
    comment_words += " ".join(tokens) + " "

# Loading the mask image
mask = np.array(Image.open(img))

# Generating the WordCloud
wordcloud = WordCloud(width=800, height=800,
                      background_color='black',
                      stopwords=stopwords,
                      min_font_size=10, mask=mask).generate(comment_words)

# Plotting the WordCloud image
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

"""## **Removing Stopwords**"""

nltk.download('stopwords')
from nltk.corpus import stopwords

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))  # To get the set of English stopwords

    if custom_stopwords:
        stop_words.update(custom_stopwords)
    return [word for word in tokens if word.lower() not in stop_words]

custom_stopwords = {'chatgpt', 'ai','artificial intelligence', 'google', 'now', 'chatbot', 'would', 'could', 'people', 'gpt',
                    'write', 'n', 'openai', 'using','prompt', 'question','one','see','will','time','got','world'}

df['Filtered_Tokens'] = df['Tokens'].apply(remove_stopwords)
df.head()

"""## **Finding 20 most common words**"""

# This is to identify and modify the list of custom stop words.
from collections import Counter

all_filtered_tokens = [word for tokens in df['Filtered_Tokens'] for word in tokens]

# Counting the occurrences of each word after removing stopwords
word_counts_after_removal = Counter(all_filtered_tokens)

# To get the 20 most common words after removing stopwords
most_common_words_after_removal = word_counts_after_removal.most_common(20)
print(most_common_words_after_removal)

import matplotlib.pyplot as plt
import seaborn as sns

words = [word for word, count in most_common_words_after_removal]
counts = [count for word, count in most_common_words_after_removal]

# Setting up the plot
plt.figure(figsize=(10, 6))

# Creating a bar plot
sns.barplot(x=counts, y=words, palette='viridis')

# Adding labels and title
plt.xlabel('Count')
plt.ylabel('Words')
plt.title('Top 20 Most Common Words after Stopwords Removal')

# Displaying the plot
plt.show()

"""## **Lemmatization**"""

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

# Initializing the lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word, pos='v') for word in tokens]

df['Lemmatized_Tokens'] = df['Filtered_Tokens'].apply(lemmatize_tokens)
df.head()

"""## **Balancing the Data**"""

import pandas as pd
from sklearn.utils import resample

# Separating majority and minority classes
df_majority = df[df['labels'] == 'bad']
df_minority = df[df['labels'] == 'good']

# Downsampling the majority class to match the minority class
df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=len(df_minority),
                                   random_state=42)

# Combining minority class with the downsampled majority class
df = pd.concat([df_majority_downsampled, df_minority])

# Shuffling the resulting dataset to mix positive and negative samples
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# To check the distribution of the balanced dataset
print(df['labels'].value_counts())

sns.countplot(x='labels', data=df)  # Visualizing the label distribution

"""## **Feature Extraction using TF-IDF Vectorization**"""

# # Performing TF-IDF vectorization
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert lemmatized tokens to a single string per row
df['TFIDF_Tokens'] = df['Lemmatized_Tokens'].apply(lambda x: ' '.join(x))

# Define the vectorizer globally so it can be saved and reused
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

# Fit and transform the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(df['TFIDF_Tokens'])

X = tfidf_matrix
y = df['labels']


"""## **Splitting the Dataset**"""

from sklearn.model_selection import train_test_split
X = tfidf_matrix
y = df['labels']

# Splitting the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""# **Model training and testing**

"""## **Support Vector Classifier**"""

# SUPPORT VECTOR CLASSIFIER
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC

model=SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Support Vector Classifier Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


# Plotting confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['good', 'bad'])

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['good', 'bad'], yticklabels=['good', 'bad'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

import pickle
import os

# Creates model folder if not exists
os.makedirs("model", exist_ok=True)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save TF-IDF vectorizer
with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

print("✅ Model and Vectorizer saved successfully.")
