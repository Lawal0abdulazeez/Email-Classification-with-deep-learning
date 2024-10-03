### **EDA ON SPAM EMAIL DATASET**

## About the Dataset
The dataset 'Spam Email' contains 2 columns, each are:
- Category: Whether it is spam or ham

- Message: context of message

## 2. Notebook Objectives

Goal of this notebook is to:

1. 📊Explore each columns' distribution in the dataset

2. 📉Analysis on ham messages

3. 📈Analysis on spam messages

## **1. Load Necessary Libraries and Dataset**
"""

# for data
import pandas as pd
import numpy as np

# for visualization
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from wordcloud import WordCloud


# nltk used for NLP and word processing
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Data Preprocessing (sklearn)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Data Modeling
from sklearn.ensemble import RandomForestClassifier
#from lightgbm.sklearn import LGBMClassifier
#import xgboost as xgb
#from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Neural Network
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Bidirectional, Conv1D, MaxPooling1D, Embedding, LSTM, GlobalMaxPooling1D, Dropout, SimpleRNN, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam


# scoring
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, RocCurveDisplay, auc

# Checking Training Time
import time

# styling
plt.style.use('ggplot')

# Loading our Dataset into a dataframe
df = pd.read_csv('spam.csv')
df.head()

# Checking Basic info of the data set
df.info()

# Checking the shape of the data
print(f"This data contains {df.shape[0]} rows and {df.shape[1]} columns. ")

# Making a visual for missing values
msno.matrix(df).set_title('Distribution of missing values',fontsize=20) # checking for missing values
print('No missing Value')

# Checking for duplicated values
df.duplicated().sum()

df = df.drop_duplicates()

"""### **2.EDA on features**

## Two parts in 'EDA on each categories' section, each are:
1. Explore Distribution of each categories

2.  Explore ham & spam message length distribution

### **2.1. Distribution of each category**
"""

category_ct = df['Category'].value_counts()

fig = px.pie(values=category_ct.values,
             names=category_ct.index,
             color_discrete_sequence=px.colors.sequential.OrRd,
             title= 'Pie Graph: spam or not')
fig.update_traces(hoverinfo='label+percent', textinfo='label+value+percent', textfont_size=15,
                  marker=dict(line=dict(color='#000000', width=2)))
fig.show()

"""### **2.2. Length distribution of spam & ham meesage**"""

# Display basic info about the DataFrame
print("DataFrame info:")
print(df.info())

# Check for missing values and types in 'text' column
print("Checking for missing or non-string values in 'Message':")
print(df['Message'].apply(type).value_counts())  # Counts of different data types in 'text'

# Replace non-string values with empty string or NaN
df['Message'] = df['Message'].astype(str)

# Compute the length of each message
df['length'] = df['Message'].apply(len)

# Separate ham and spam messages
ham = df[df['Category'].str.strip().str.lower() == 'ham']
spam = df[df['Category'].str.strip().str.lower() == 'spam']

# Print data to confirm
print("Ham DataFrame:")
print(ham.head())
print("Spam DataFrame:")
print(spam.head())

# Check lengths of the messages
print("Ham Lengths:")
print(ham['length'].describe())
print("Spam Lengths:")
print(spam['length'].describe())

# Data for distribution plot
if not ham.empty and not spam.empty:
    hist_data = [ham['length'].tolist(), spam['length'].tolist()]
    group_labels = ['ham', 'spam']
    colors = ['black', 'red']

    # Create distribution plot
    fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)

    # Add title
    fig.update_layout(title_text='Length Distribution of Ham and Spam Messages',
                      template='simple_white')

    # Show plot
    fig.show()
else:
    print("No data available for plotting.")

"""Spam messages are mainly distributed right on 100 while Ham messages are distributed left on the length of 100.
Thus, we can conclude as spam message tends to have more letters than hpam message.

#### Rank of Ham Terms
"""

# Separate ham messages
ham = df[df['Category'].str.strip().str.lower() == 'ham']

# Tokenize and count terms
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(ham['Message'])
term_freq = X.sum(axis=0).A1
terms = vectorizer.get_feature_names_out()

# Create a DataFrame with term frequencies
freq_df = pd.DataFrame({'Term': terms, 'Frequency': term_freq})
freq_df = freq_df.sort_values(by='Frequency', ascending=False)

# Select the top 10 terms
top_10 = freq_df.head(10)

# Plot using Plotly Express
fig = px.bar(top_10, x='Term', y='Frequency', text='Frequency', color='Term',
             color_discrete_sequence=px.colors.sequential.PuBuGn,
             title='Rank of Ham Terms',
             template="simple_white")

# Customize plot
for idx in range(len(top_10)):
    fig.data[idx].marker.line.width = 2
    fig.data[idx].marker.line.color = "black"

fig.update_traces(textposition='inside',
                  textfont_size=11)

# Show plot
fig.show()

"""#### World Cloud of Ham Messages"""

# Separate ham messages
ham = df[df['Category'].str.strip().str.lower() == 'ham']

# Tokenize and count terms
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(ham['Message'])
term_freq = X.sum(axis=0).A1
terms = vectorizer.get_feature_names_out()

# Create a DataFrame with term frequencies
freq_df = pd.DataFrame({'Term': terms, 'Frequency': term_freq})
freq_df = freq_df.sort_values(by='Frequency', ascending=False)

# Prepare data for WordCloud
data = dict(zip(freq_df['Term'], freq_df['Frequency']))

# Create and display the Word Cloud
wordcloud = WordCloud(width=800, height=400,
                      background_color='white',
                      colormap='seismic',
                      contour_color='black',
                      contour_width=1).generate_from_frequencies(data)

# Plot the Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Terms of Ham Messages', fontsize=15)
plt.show()

"""#### Bar chart of Spam Messages"""

# Separate spam messages
spam = df[df['Category'].str.strip().str.lower() == 'spam']

# Tokenize and count terms
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(spam['Message'])
term_freq = X.sum(axis=0).A1
terms = vectorizer.get_feature_names_out()

# Create a DataFrame with term frequencies
freq_df = pd.DataFrame({'Term': terms, 'Frequency': term_freq})
freq_df = freq_df.sort_values(by='Frequency', ascending=False)

# Get the top 10 terms
top_10 = freq_df.head(10)

# Plot using plotly.express
fig = px.bar(top_10, x='Term', y='Frequency', text='Frequency',
             color='Term',
             color_discrete_sequence=px.colors.sequential.PuRd,
             title='Rank of Spam Terms',
             template='simple_white')

# Add black outline to bars
for idx in range(len(top_10)):
    fig.data[idx].marker.line.width = 2
    fig.data[idx].marker.line.color = "black"

fig.show()

"""#### WordCloud of Spam Messages"""

# Separate spam messages
spam = df[df['Category'].str.strip().str.lower() == 'spam']

# Tokenize and count terms
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(spam['Message'])
term_freq = X.sum(axis=0).A1
terms = vectorizer.get_feature_names_out()

# Create a DataFrame with term frequencies
freq_df = pd.DataFrame({'Term': terms, 'Frequency': term_freq})
freq_df = freq_df.sort_values(by='Frequency', ascending=False)

# Convert to dictionary for word cloud
data = dict(zip(freq_df['Term'], freq_df['Frequency']))

# Generate word cloud
spam_wordcloud = WordCloud(background_color='white',
                          colormap='seismic',
                          width=800,
                          height=400,
                          max_words=200).generate_from_frequencies(data)

# Plot word cloud
plt.figure(figsize=(10, 5))
plt.imshow(spam_wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Terms of Spam Messages')
plt.show()

dataset_info = {
    "Number of Rows": len(df),
    "Number of Columns": df.shape[1],
    "Missing Values": df.isnull().sum(),
    "Spam Distribution": df['Category'].value_counts(normalize=True) * 100
}

dataset_info



"""# **4. Text preprocessing for spam email detection**

### **Preprocess dataframe for classification in the next section**

function to get words from sentence, and lemmatize it with removing stopwords.
"""

def preprocess(sentence):
    words = get_word(sentence)
    words_ltz = lemmatization(words)
    removed = remove_stopword('1',words_ltz)
    return removed

"""and also replace 'ham' value into 1, 'spam' value into 0."""

df.head()

# Preprocessing
texts = df['Message'].values
labels = df['Category'].values

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Assuming your dataset has the features in 'X' and labels in 'y'
texts = df['Message']  # Features (you may need to vectorize text data first)
labels = df['Category']  # Target

# Convert X into a numerical format (like using TF-IDF or CountVectorizer)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
texts = vectorizer.fit_transform(texts)

# Apply random undersampling
rus = RandomUnderSampler(random_state=42)
texts_u, labels_u = rus.fit_resample(texts, labels)

# Import the LabelEncoder class
from sklearn.preprocessing import LabelEncoder

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)



"""## **Data Processing using Sequence Based Method**"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import EditedNearestNeighbours

# Preprocessing
texts = df['Message'].values
labels = df['Category'].values

# Convert texts into numerical format using CountVectorizer
vectorizer = CountVectorizer()
texts_vectorized = vectorizer.fit_transform(texts)

# Apply NearMiss undersampling (you can change the version, e.g., version=2)
near_miss = NearMiss(version=1)
texts_resampled, labels_resampled = near_miss.fit_resample(texts_vectorized, labels)


# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_resampled)

# Convert sparse matrix to dense format before train-test split
texts_resampled_dense = texts_resampled.toarray()

# Split the dataset (use the resampled texts and labels)
X_train, X_test, y_train, y_test = train_test_split(
    texts_resampled_dense, labels_encoded, test_size=0.2, random_state=42
)

# Tokenization and Padding
max_words = 5000  # Limit the number of words in the tokenizer
max_len = 150  # Max length of sequences

# Convert X_train and X_test from arrays to lists of strings
X_train_list = [str(item) for item in X_train.tolist()]
X_test_list = [str(item) for item in X_test.tolist()]

# Tokenizer for training data
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_list)  # Fit tokenizer on training data

# Convert texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train_list)
X_test_seq = tokenizer.texts_to_sequences(X_test_list)

# Padding the sequences to ensure uniform input size
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Create a DataFrame for easy plotting
data = pd.DataFrame({
    'Set': ['Training'] * len(y_train) + ['Testing'] * len(y_test),
    'Label': list(y_train) + list(y_test)
})
# Define colors
color_map = {0: '#1f77b4', 1: '#ff7f0e'}  # Map colors to label
# Plot histograms
plt.figure(figsize=(12, 6))

# Training Set Histogram
plt.subplot(1, 2, 1)
sns.countplot(x='Label', data=data[data['Set'] == 'Training'], hue='Label', palette=color_map)
plt.title('Training Set Label Distribution')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks(ticks=[0, 1], labels=['ham (0)', 'spam (1)'])
plt.legend(title='Label', labels=['ham (0)', 'spam (1)'], loc='upper right')

# Testing Set Histogram
plt.subplot(1, 2, 2)
sns.countplot(x='Label', data=data[data['Set'] == 'Testing'], hue='Label', palette=color_map)
plt.title('Testing Set Label Distribution')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks(ticks=[0, 1], labels=['ham (0)', 'spam (1)'])
plt.legend(title='Label', labels=['ham (0)', 'spam (1)'], loc='upper right')

plt.tight_layout()
plt.show()

import pydot
import graphviz

import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

import pydot

try:
    # Attempt to create a simple dot graph
    graph = pydot.graph_from_dot_data("digraph G {A -> B;}")[0]
    graph.write_png("test_graph.png")  # Try to write to a PNG file
    print("Graphviz is working correctly.")
except Exception as e:
    print("Error:", e)

"""## **Advance Individual Model**

##### CNN Model
"""

import pydot

# Parameters for the model
max_words = 5000  # Adjust this based on your vocabulary size

# Define the CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128))  # Removed input_length
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the CNN model instance
model_cnn = create_cnn_model()

# Build the model by specifying the input shape
model_cnn.build(input_shape=(None, max_words))

# Print the model summary
print(model_cnn.summary())

# Show the model architecture with a smaller size
tf.keras.utils.plot_model(
    model_cnn,
    show_shapes=True,
    show_layer_names=True,
    dpi=60  # Adjust DPI for smaller image size
)

"""##### RNN Model"""

# Parameters for the model
max_words = 5000  # Adjust this based on your vocabulary size

# Define the RNN model
def create_rnn_model():
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128))  # Removed input_length
    model.add(SimpleRNN(128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the RNN model instance
model_rnn = create_rnn_model()

# Build the model by specifying the input shape
model_rnn.build(input_shape=(None, max_words))

# Print the model summary
print(model_rnn.summary())

# Show the model architecture with a smaller size
tf.keras.utils.plot_model(
    model_rnn,          # Corrected variable name
    show_shapes=True,
    show_layer_names=True,
    dpi=60  # Adjust DPI for smaller image size
)

"""##### LSTM Model"""

# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
import tensorflow as tf

# Parameters for the model
max_words = 5000  # Adjust this based on your vocabulary size

# Define the LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=128))  # Removed input_length
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the LSTM model instance
model_lstm = create_lstm_model()

# Build the model by specifying the input shape
model_lstm.build(input_shape=(None, max_words))

# Print the model summary
print(model_lstm.summary())

# Show the model architecture with a smaller size
tf.keras.utils.plot_model(
    model_lstm,
    show_shapes=True,
    show_layer_names=True,
    dpi=60  # Adjust DPI for smaller image size
)



"""### **Evaluation of Individual Model**"""

# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping

# Define combined models (ensure model creation functions are implemented)
combined_models_1 = {
    'CNN': create_cnn_model(),
    'RNN': create_rnn_model(),
    'LSTM': create_lstm_model()
}

class_labels = ['ham', 'spam']

# Dictionary to store metrics and history of each model
metric_combine = {}
history_combine = {}

# Initialize a dictionary to store classification reports and confusion matrices
reports = {}

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

for name, model in combined_models_1.items():
    print(f"Training {name} model...")

    # Train the model with EarlyStopping callback and save the training history
    history = model.fit(
        X_train_pad,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=2,
        callbacks=[early_stopping]  # Add the callback here
    )
    history_combine[name] = history.history  # Save history

    # Evaluate the model on test data
    y_pred = (model.predict(X_test_pad) > 0.5).astype("int32").flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)


    metric_combine[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_labels)
    reports[name] = report

    print(f"{name} model metrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\n")
    print(f"Classification Report for {name}:\n", report)

# Plot training history for each model
for name, history in history_combine.items():
    plt.figure(figsize=(12, 6))

    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.suptitle(f'Training History for {name}')
    plt.show()

# Convert metrics dictionary to a DataFrame for easier plotting
metrics_df = pd.DataFrame(metric_combine).T

# Define plot size
plt.figure(figsize=(8, 8))

# Define bar width and positions
bar_width = 0.1
index = np.arange(len(metrics_df))

# Plot each metric
for i, metric in enumerate(metrics_df.columns):
    bars = plt.bar(index + i * bar_width, metrics_df[metric] * 100, bar_width,
                   label=metric, color=['blue', 'green', 'red', 'orange'][i])

    # Annotate bars with the exact percentage value
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}%',
                 ha='center', va='bottom', fontsize=9, rotation=45, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Adding labels and title
plt.xlabel('Models')
plt.ylabel('Percentage (%)')
plt.title('Comparison of Evaluation Metrics for Single Models')

# Adjust x-axis labels to avoid overlap
plt.xticks(index + bar_width * (len(metrics_df.columns) / 2) - bar_width / 2, metrics_df.index, rotation=45)

# Adjust plot margins to ensure everything fits
plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust the margins

plt.legend(loc='lower left')
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Prepare the figure
plt.figure(figsize=(10, 8))

# Iterate over each model to compute and plot ROC curve
for model_name, model in combined_models_1.items():
    # Predict probabilities
    y_proba = model.predict(X_test_pad).flatten()

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')

# Plot the diagonal line (no redundant arguments)
plt.plot([0, 1], [0, 1], 'r--')

# Plot settings
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.grid(False)

# Show the plot
plt.tight_layout()
plt.show()

"""## **Hybrid Models Considered**

### **CNN-RNN Model**
"""

# Import necessary libraries
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, SimpleRNN, Dropout, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

# Parameters for the model
max_words = 5000  # Adjust this based on your vocabulary size
max_len = 100     # Adjust this based on the length of your input sequences

# Define the CNN-RNN model
def create_cnn_rnn_model():
    input_layer = Input(shape=(max_len,))
    x = Embedding(input_dim=max_words, output_dim=128)(input_layer)
    x = Conv1D(filters=64, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = SimpleRNN(128, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='relu')(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the CNN-RNN model instance
model_cnn_rnn = create_cnn_rnn_model()

# Print the model summary
print(model_cnn_rnn.summary())

# Show the model architecture
tf.keras.utils.plot_model(
    model_cnn_rnn,
    show_shapes=True,
    show_layer_names=True,
    dpi=60  # Adjust DPI for smaller image size
)

"""### **CNN-LSTM model**"""

# Import necessary libraries
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

# Parameters for the model
max_words = 5000  # Adjust this based on your vocabulary size
max_len = 150     # Adjust this based on the length of your input sequences

# Define the CNN-LSTM model
def create_cnn_lstm_model():
    input_layer = Input(shape=(max_len,))
    x = Embedding(input_dim=max_words, output_dim=128)(input_layer)
    x = Conv1D(filters=64, kernel_size=5, activation='relu')(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='relu')(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the CNN-LSTM model instance
model_cnn_lstm = create_cnn_lstm_model()

# Print the model summary
print(model_cnn_lstm.summary())

# Show the model architecture
tf.keras.utils.plot_model(
    model_cnn_lstm,
    show_shapes=True,
    show_layer_names=True,
    dpi=60  # Adjust DPI for smaller image size
)

"""## **RNN-LSTM Model**"""

# Import necessary libraries
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
import tensorflow as tf

# Parameters for the model
max_words = 5000  # Adjust this based on your vocabulary size
max_len = 150     # Adjust this based on the length of your input sequences

# Define the RNN-LSTM model
def create_rnn_lstm_model():
    input_layer = Input(shape=(max_len,))
    x = Embedding(input_dim=max_words, output_dim=128)(input_layer)
    x = SimpleRNN(128, return_sequences=True)(x)
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='relu')(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create the RNN-LSTM model instance
model_rnn_lstm = create_rnn_lstm_model()

# Print the model summary
print(model_rnn_lstm.summary())

# Show the model architecture
tf.keras.utils.plot_model(
    model_rnn_lstm,
    show_shapes=True,
    show_layer_names=True,
    dpi=60  # Adjust DPI for smaller image size
)

"""# **Evaluation of Each Combination**"""

# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping

# Define combined models (ensure model creation functions are implemented)
combined_models_2 = {
    'CNN-RNN': create_cnn_rnn_model(),
    'CNN-LSTM': create_cnn_lstm_model(),
    'RNN-LSTM': create_rnn_lstm_model()
}

class_labels = ['ham', 'spam']


# Initialize a dictionary to store classification reports and confusion matrices
reports = {}

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

for name, model in combined_models_2.items():
    print(f"Training {name} model...")

    # Train the model with EarlyStopping callback and save the training history
    history = model.fit(
        X_train_pad,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=2,
        callbacks=[early_stopping]  # Add the callback here
    )
    history_combine[name] = history.history  # Save history

    # Evaluate the model on test data
    y_pred = (model.predict(X_test_pad) > 0.5).astype("int32").flatten()

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)


    metric_combine[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Generate the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=class_labels)
    reports[name] = report

    print(f"{name} model metrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\n")
    print(f"Classification Report for {name}:\n", report)

# Plot training history for each model
for name, history in history_combine.items():
    plt.figure(figsize=(12, 6))

    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.suptitle(f'Training History for {name}')
    plt.show()

# Convert metrics dictionary to a DataFrame for easier plotting
metrics_df = pd.DataFrame(metric_combine).T.iloc[3:]

# Define plot size
plt.figure(figsize=(8, 8))

# Define bar width and positions
bar_width = 0.1
index = np.arange(len(metrics_df))

# Plot each metric
for i, metric in enumerate(metrics_df.columns):
    bars = plt.bar(index + i * bar_width, metrics_df[metric] * 100, bar_width,
                   label=metric, color=['blue', 'green', 'red', 'orange'][i])

    # Annotate bars with the exact percentage value
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}%',
                 ha='center', va='bottom', fontsize=9, rotation=45, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Adding labels and title
plt.xlabel('Models')
plt.ylabel('Percentage (%)')
plt.title('Comparison of Evaluation Metrics for Combined Models')

# Adjust x-axis labels to avoid overlap
plt.xticks(index + bar_width * (len(metrics_df.columns) / 2) - bar_width / 2, metrics_df.index, rotation=45)

# Adjust plot margins to ensure everything fits
plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust the margins

plt.legend(loc='best')
plt.show()

"""##### Roc of Best Hybrid"""

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Prepare the figure
plt.figure(figsize=(10, 8))

# Iterate over each model to compute and plot ROC curve
for model_name, model in combined_models_2.items():
    # Predict probabilities
    y_proba = model.predict(X_test_pad).flatten()

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')

# Plot the diagonal line (no redundant arguments)
plt.plot([0, 1], [0, 1], 'r--')

# Plot settings
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc='lower right')
plt.grid(False)

# Show the plot
plt.tight_layout()
plt.show()

"""## **Comparison of All Models**"""

import numpy as np
import matplotlib.pyplot as plt

# Metrics for CNN, RNN, LSTM
cnn_metrics = metric_combine.get('CNN', {})
rnn_metrics = metric_combine.get('RNN', {})
lstm_metrics = metric_combine.get('LSTM', {})

# Combine metrics with combined models
combined_cnn_rnn_metrics = metric_combine.get('CNN-RNN', {})
combined_cnn_lstm_metrics = metric_combine.get('CNN-LSTM', {})
combined_rnn_lstm_metrics = metric_combine.get('RNN-LSTM', {})

# All metrics to compare
models = ['CNN', 'RNN', 'LSTM', 'CNN-RNN', 'CNN-LSTM', 'RNN-LSTM']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Create lists for metric values
metric_values = {metric: [] for metric in metrics}

# Populate metric values for each model
for model in models:
    if model == 'CNN':
        metrics_data = cnn_metrics
    elif model == 'RNN':
        metrics_data = rnn_metrics
    elif model == 'LSTM':
        metrics_data = lstm_metrics
    elif model == 'CNN-RNN':
        metrics_data = combined_cnn_rnn_metrics
    elif model == 'CNN-LSTM':
        metrics_data = combined_cnn_lstm_metrics
    elif model == 'RNN-LSTM':
        metrics_data = combined_rnn_lstm_metrics

    for metric in metrics:
        metric_values[metric].append(metrics_data.get(metric, 0))

# Set up bar width and positions
bar_width = 0.15
bar_positions = np.arange(len(models))

# Plot bars for each metric
plt.figure(figsize=(16, 8))
for i, metric in enumerate(metrics):
    bars = plt.bar(
        bar_positions + i * bar_width,
        metric_values[metric],
        bar_width,
        label=metric
    )

    # Annotate each bar with its value
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # X position (center of the bar)
            yval,                               # Y position (height of the bar)
            f'{yval:.2f}',                     # Value formatted to 2 decimal places
            ha='center',                        # Horizontal alignment
            va='bottom'                         # Vertical alignment (above the bar)
        )

# Add labels, title, and legend
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Comparison of Different Models Across Metrics')
plt.xticks(bar_positions + bar_width * (len(metrics) / 2 - 0.5), models)
plt.legend()

# Show plot
plt.tight_layout()
plt.show()

"""
### **Patterns of Misclassified Emails**

"""

def display_misclassified_emails(misclassified_indices, X_test_pad, y_test, y_pred):
    for idx in misclassified_indices[:10]:  # Adjust the slice to control how many emails you want to display
        email_content = X_test_pad[idx][:10]  # Display the first 50 tokens
        true_label = y_test[idx]
        predicted_label = y_pred[idx]

        print(f"Email (first 50 tokens): {email_content}")
        print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
        print("\n---\n")

# For RNN Model

y_pred = model_rnn.predict(X_test_pad)

# Compare y_test with the first column of y_pred since it is a 2D array
misclassified_indices = y_test != y_pred[:,0]
misclassified_emails = X_test_pad[misclassified_indices]
misclassified_labels = y_test[misclassified_indices]
misclassified_predictions = y_pred[misclassified_indices]

for email, true_label, predicted_label in zip(misclassified_emails, misclassified_labels, misclassified_predictions):
    print(f"Email: {email}")
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    print("\n")

# Get indices of misclassified emails
misclassified_indices = np.where(y_test != y_pred)[0]

# Display a sample of misclassified emails
display_misclassified_emails(misclassified_indices, X_test_pad, y_test, y_pred)

# For LSTM Model

y_pred = model_lstm.predict(X_test_pad)

# Compare y_test with the first column of y_pred since it is a 2D array
misclassified_indices = y_test != y_pred[:,0]
misclassified_emails = X_test_pad[misclassified_indices]
misclassified_labels = y_test[misclassified_indices]
misclassified_predictions = y_pred[misclassified_indices]

for email, true_label, predicted_label in zip(misclassified_emails, misclassified_labels, misclassified_predictions):
    print(f"Email: {email}")
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    print("\n")

# Get indices of misclassified emails
misclassified_indices = np.where(y_test != y_pred)[0]

# Display a sample of misclassified emails
display_misclassified_emails(misclassified_indices, X_test_pad, y_test, y_pred)

# Assuming 'model_rnn_lstm' is your trained RNN-LSTM model object
y_pred = model_rnn_lstm.predict(X_test_pad)

# Compare y_test with the first column of y_pred since it is a 2D array
misclassified_indices = y_test != y_pred[:,0]
misclassified_emails = X_test_pad[misclassified_indices]
misclassified_labels = y_test[misclassified_indices]
misclassified_predictions = y_pred[misclassified_indices]

for email, true_label, predicted_label in zip(misclassified_emails, misclassified_labels, misclassified_predictions):
    print(f"Email: {email}")
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    print("\n")

# Get indices of misclassified emails
misclassified_indices = np.where(y_test != y_pred)[0]

# Display a sample of misclassified emails
display_misclassified_emails(misclassified_indices, X_test_pad, y_test, y_pred)

# For CNN-LSTM Model

y_pred = model_cnn_lstm.predict(X_test_pad)

# Compare y_test with the first column of y_pred since it is a 2D array
misclassified_indices = y_test != y_pred[:,0]
misclassified_emails = X_test_pad[misclassified_indices]
misclassified_labels = y_test[misclassified_indices]
misclassified_predictions = y_pred[misclassified_indices]

for email, true_label, predicted_label in zip(misclassified_emails, misclassified_labels, misclassified_predictions):
    print(f"Email: {email}")
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    print("\n")

# Get indices of misclassified emails
misclassified_indices = np.where(y_test != y_pred)[0]

# Display a sample of misclassified emails
display_misclassified_emails(misclassified_indices, X_test_pad, y_test, y_pred)

X_test_pad.shape[1]

from tensorflow.keras.models import Sequential, load_model

# Save the model
model_cnn_rnn.save('cnn_rnn.h5')
print("Model saved as cnn_rnn.h5")

# Load the model
loaded_model = load_model('cnn_rnn.h5')
print("Model loaded from cnn_rnn.h5")

# Prepare input text for prediction
texts_to_predict = ["FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, Â£1.50 to rcv",
                    "Even my brother is not like to speak with me. They treat me like aids patent."]
texts_seq = tokenizer.texts_to_sequences(texts_to_predict)
texts_pad = pad_sequences(texts_seq, maxlen=100, padding='post')

# Make predictions
y_pred = loaded_model.predict(texts_pad)
predicted_labels = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels

# Display predictions
for text, label in zip(texts_to_predict, predicted_labels):
    print(f"Text: {text}, Predicted Label: {label[0]}")