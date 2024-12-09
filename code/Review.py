"""
Author: Jay A. Panchal & Jil P. Makwana
Description: This project aims to detect fake reviews from e-commerce websites using
machine learning models by processing, cleaning, and extracting features from textual data to classify reviews as FAKE or REAL.
"""

"--------------------------------------------------------------------------------------------------------------------"

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
import warnings
import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score
warnings.filterwarnings("ignore", category=RuntimeWarning)

"--------------------------------------------------------------------------------------------------------------------"

# Visualizing the distribution of target classes (Fake/Real)
file_path = r'.\data\fake reviews dataset.csv'  

if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("Data loaded successfully!")  
else:
    print(f"File not found: {file_path}")


warnings.filterwarnings("ignore", category=FutureWarning)


plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='label', palette='Set2', legend=False)
plt.title('distribution of target classes')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()


# Check for missing values and class distribution
print("Checking for Missing Values and Class Distribution: ")
print(data.isnull().sum())
print(data['label'].value_counts())

"--------------------------------------------------------------------------------------------------------------------"

# Data cleaning function to remove HTML tags, punctuation, and convert text to lowercase
import re

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Supprimer les balises HTML
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer les caractères spéciaux
    text = text.lower()  # Transformer en minuscules
    return text

data['cleaned_text'] = data['text_'].apply(clean_text)
print("The data has been successfully cleaned and stored in the 'cleaned_text' column.")
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

"--------------------------------------------------------------------------------------------------------------------"

# Applying cleaning function to the dataset
data['text_length'] = data['cleaned_text'].apply(len)

# Plot the distribution of text lengths
plt.figure(figsize=(10, 6))
sns.histplot(data['text_length'], bins=30, kde=True)
plt.title('text length distribution')
plt.xlabel('Length of the text')
plt.ylabel('frequency')
plt.show()

"--------------------------------------------------------------------------------------------------------------------"

# Tokenizing the text by removing stopwords (common words that don't add much value)
nltk.download('stopwords', quiet=True) 
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words] 
    return tokens

# Applying tokenization to the cleaned text
data['tokens'] = data['cleaned_text'].apply(tokenize)

"--------------------------------------------------------------------------------------------------------------------"

# Visualizing most frequent words in FAKE and REAL reviews
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_top_words(data, category):
    tokens = [token for tokens_list in data[data['label'] == category]['tokens'] for token in tokens_list]
    word_counts = Counter(tokens)
    most_common = word_counts.most_common(10)
    
    words, counts = zip(*most_common)
    df_words = pd.DataFrame({'word': words, 'count': counts})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_words, x='count', y='word', palette='Blues_d')
    plt.title(f'Most Frequent in Category {category}')
    plt.xlabel('frequency')
    plt.ylabel('Words')
    plt.show()

# Ploting for both FAKE and REAL reviews
for label in data['label'].unique():
    plot_top_words(data, label)

"--------------------------------------------------------------------------------------------------------------------"

# Feature extraction using TF-IDF Vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
X = vectorizer.fit_transform(data['cleaned_text']).toarray()  # Convert text to feature vectors
y = data['label']  # Target variable (FAKE or REAL)

# Spliting dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"--------------------------------------------------------------------------------------------------------------------"

# Ploting confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

"--------------------------------------------------------------------------------------------------------------------"

"""
Here we will Training Multinomial Naive Bayes classifier!!!
Naive Bayes is a probabilistic classifier based on Bayes' Theorem, assuming feature independence, and classifies 
data by selecting the class with the highest probability, commonly used for text classification like spam detection 
and sentiment analysis.
"""
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
report = classification_report(y_test, pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Report for Naive Bayes Classifier")
print(report_df)
print("Accuracy for Naive Bayes Classifier:   %0.3f" % score)
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
# Confusion matrix for Multinomial Naive Bayes
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'], normalize=False)
plt.show()

"--------------------------------------------------------------------------------------------------------------------"

"""
Training Passive Aggressive Classifier!! 
The Passive Aggressive Classifier is an online learning algorithm that updates its model aggressively when it 
misclassifies and passively when it’s correct, making it effective for large-scale and streaming data tasks.
"""

from sklearn.linear_model import PassiveAggressiveClassifier

linear_clf = PassiveAggressiveClassifier(max_iter=100)
linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
report = classification_report(y_test, pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Report for Passive Aggressive  Classifier")
print(report_df)
print("Accuracy for Passive Aggressive Classifier:   %0.3f" % score)
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")



# Confusion matrix for Passive Aggressive Classifier
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])

"--------------------------------------------------------------------------------------------------------------------"

# Tuning the Naive Bayes classifier using different alpha values
previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
"--------------------------------------------------------------------------------------------------------------------"

# Feature importance and sentence analysis
feature_names = vectorizer.get_feature_names_out()
feature_log_prob = classifier.feature_log_prob_

top_positive_idx = feature_log_prob[1].argsort()[-10:][::-1] 
top_negative_idx = feature_log_prob[1].argsort()[:10] 

# Extract top positive and negative words
feature_names = vectorizer.get_feature_names_out()
top_positive_words = feature_names[top_positive_idx]
top_negative_words = feature_names[top_negative_idx]

# Function to get sentences containing important words
def get_sentences_with_keywords(words, sentences):
    sentences_with_keywords = []
    for sentence in sentences:
        if any(word in sentence for word in words):
            sentences_with_keywords.append(sentence)
    return sentences_with_keywords

# Find sentences with top positive and negative words
top_positive_sentences = get_sentences_with_keywords(top_positive_words, data['cleaned_text'])
top_negative_sentences = get_sentences_with_keywords(top_negative_words, data['cleaned_text'])

# Count occurrences of positive and negative words in these sentences
positive_counts = {word: sum(word in sentence for sentence in top_positive_sentences) for word in top_positive_words}
negative_counts = {word: sum(word in sentence for sentence in top_negative_sentences) for word in top_negative_words}

# Convert counts to DataFrame for plotting
positive_df = pd.DataFrame(list(positive_counts.items()), columns=['Word', 'Count'])
negative_df = pd.DataFrame(list(negative_counts.items()), columns=['Word', 'Count'])

# Plot the bar graphs for positive and negative words
plt.figure(figsize=(14, 6))

# Positive words graph
plt.subplot(1, 2, 1)
sns.barplot(data=positive_df, x='Count', y='Word', palette='Greens_d')
plt.title('Frequency of Sentences with Positive Review')
plt.xlabel('Sentence Count')
plt.ylabel('Positive Words')

# Negative words graph
plt.subplot(1, 2, 2)
sns.barplot(data=negative_df, x='Count', y='Word', palette='Reds_d')
plt.title('Frequency of Sentences with Negative Reviews')
plt.xlabel('Sentence Count')
plt.ylabel('Negative Words')

plt.tight_layout()
plt.show()


"--------------------------------------------------------------------------------------------------------------------"
"""
This we will use for voting classifer (using multiple classifiers) with soft voting!!!!
This code trains a Voting Classifier with Multinomial Naive Bayes, 
Logistic Regression, and Random Forest using soft voting, then evaluates accuracy and 
precision for the 'OR' class, with a Random Forest as the final estimator for a Stacking 
Classifier to improve prediction performance.
"""

mnb = MultinomialNB()
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)

voting = VotingClassifier(estimators=[('LR', lrc), ('nb', mnb), ('RF', rfc)],voting='soft')

# Fit the model and evaluate performance
voting.fit(X_train,y_train)

y_pred = voting.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Report for Voting Classifier")
print(report_df)
print("Accuracy of the model (Voting classifier)",accuracy_score(y_test,y_pred))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

estimators = [('LR', lrc), ('nb', mnb), ('RF', rfc)]
final_estimator = RandomForestClassifier(n_estimators=50, random_state=2)


"--------------------------------------------------------------------------------------------------------------------"

"""
A stacking classifier combines multiple base models (estimators) whose predictions are 
merged by a final estimator, and estimators is a list of tuples, each containing a model's
name and object, e.g., estimators = [('lr', LogisticRegression()), ('rf', RandomForestClassifier()), 
('nb', MultinomialNB())].
Stacking Classifier - Combining multiple base models and using a final estimator
"""
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print("Report for Voting Classifier")
print(report_df)
print("Accuracy of the model(Stacking classifier)",accuracy_score(y_test,y_pred))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


"--------------------------------------------------------------------------------------------------------------------"

# Save the trained models (vectorizer and classifier) using pickle
pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))
"--------------------------------------------------------------------------------------------------------------------"

print("🚀 Mission Accomplished: Task Complete! 🎉💥")