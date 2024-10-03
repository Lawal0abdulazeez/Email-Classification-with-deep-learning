# Email Spam/Ham Classifier

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
  - [Dataset Preparation](#dataset-preparation)
  - [Preprocessing](#preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training the Model](#training-the-model)
  - [Saving the Model](#saving-the-model)
- [Application Development](#application-development)
  - [Streamlit App Structure](#streamlit-app-structure)
  - [Preprocessing Function](#preprocessing-function)
- [Deployment](#deployment)
  - [Local Deployment](#local-deployment)
  - [Cloud Deployment](#cloud-deployment)
- [Project Details](#project-details)
- [Testing](#testing)
- [About the Author](#about-the-author)
- [License](#license)

## Project Overview

The **Email Spam/Ham Classifier** is a Deep Learning application that leverages a Convolutional Neural Network (CNN) combined with a Recurrent Neural Network (RNN) and LSTM to classify emails as either spam or ham (non-spam). The main goal is to provide a user-friendly tool that can effectively filter unwanted emails, helping users maintain a clean and organized inbox. This project showcases the implementation of a deep learning model in a web application using Streamlit, allowing for real-time predictions based on user input.

## Technologies Used

The project utilizes several key technologies:

- **Python**: The primary programming language used for developing the application.
- **TensorFlow**: A popular deep learning library that facilitates building and training the neural network model.
- **Keras**: An API integrated with TensorFlow, simplifying the creation of neural networks.
- **Streamlit**: A web framework that allows for the rapid development of web applications directly from Python scripts.
- **NumPy**: A fundamental package for numerical computing in Python, used for handling data arrays.
- **Pickle**: A Python library for object serialization, utilized to save the tokenizer object after training.

## Installation

To set up the project locally, follow these instructions:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Lawal0abdulazeez/Email-Classification-with-deep-learning
   cd Email-Classification-with-deep-learning
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install Required Libraries**:
   Ensure you have a `requirements.txt` file listing all dependencies. Install them using:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Once the project is set up, you can run the application as follows:

1. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the Application**:
   Open your web browser and navigate to `http://localhost:8501`.

3. **Classify Emails**:
   - Input the content of the email in the provided text area.
   - Click the **Predict** button to classify the email as **Spam** or **Ham**.

## Model Training

### Dataset Preparation

The dataset for training the model must contain labeled examples of emails categorized as spam and ham. You can use publicly available datasets, such as:

- **SpamAssassin Public Corpus**: Contains a diverse collection of spam and non-spam emails.
- **Enron Email Dataset**: A set of emails from the Enron Corporation, including various topics and contexts.

Make sure to clean the dataset to remove any unnecessary information and ensure the emails are formatted consistently.

### Preprocessing

In the preprocessing step, the text data is cleaned, tokenized, and padded:

1. **Text Cleaning**: 
   - Remove special characters, URLs, and unnecessary whitespace to normalize the text. This helps improve the model's performance.

2. **Tokenization**:
   - Convert the cleaned text into sequences of integers. Each unique word is assigned an integer ID based on its frequency.

3. **Padding**:
   - Since neural networks require fixed-size input, pad the sequences to a uniform length using the `pad_sequences` function. This ensures that all input data has the same shape.

Here’s a code snippet that illustrates these preprocessing steps:
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(email_texts)  # email_texts is a list containing the cleaned email content

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(email_texts)
maxlen = 150  # Adjust maxlen according to the dataset
padded_sequences = pad_sequences(sequences, maxlen=maxlen)
```

### Model Architecture

The model architecture consists of several layers:

- **Embedding Layer**: This layer converts the integer sequences into dense vector representations. It captures the semantic meaning of the words.
- **Convolutional Layer**: It applies convolution operations on the embedded input to extract local features. The kernel size and number of filters can be adjusted based on the dataset.
- **MaxPooling Layer**: Reduces the dimensionality of the feature maps and helps to avoid overfitting by retaining the most important features.
- **Recurrent Layer (LSTM/GRU)**: This layer processes the sequence data, capturing the temporal relationships within the email text.
- **Dense Layer**: The final layer outputs a single value between 0 and 1, indicating the probability of the email being spam.

Here's an example of the model architecture:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
```

### Training the Model

Compile and train the model using the prepared training data. You can adjust parameters such as the number of epochs and batch size based on the training performance:
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded_sequences, labels, epochs=5, batch_size=32, validation_split=0.2)
```

### Saving the Model

After training, save the model and tokenizer for future use:
```python
model.save('cnn_lstm.h5')

# Save the tokenizer
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

## Application Development

### Streamlit App Structure

The main file for the application is `app.py`. It contains:

- **Title and Description**: Provides users with an overview of what the application does.
- **Input Section**: Where users can input the content of an email for classification.
- **Prediction Button**: Triggers the model to classify the input text.
- **Output Section**: Displays the prediction result, indicating whether the email is spam or ham.

### Preprocessing Function

A dedicated function processes the user input text before passing it to the model for prediction:
```python
def preprocess_input(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=150)  # Ensure consistent input size
    return padded_sequences
```
This function ensures that the text is in the correct format before prediction.

## Deployment

### Local Deployment

To run the application locally, simply execute the Streamlit command in your terminal:
```bash
streamlit run app.py
```
This will host the application on your local server, typically accessible at `http://localhost:8501`.

### Cloud Deployment

For cloud deployment, consider using platforms such as:

- **Streamlit Sharing**: Deploy your app easily by pushing your code to GitHub and sharing it through Streamlit's platform. Follow the [Streamlit Sharing documentation](https://share.streamlit.io/) for guidance.

## Project Details

The project contains the following files:
- `emailspam.ipynb`: It contains the code for Data processing and model Training
- `app.py`: The main Streamlit application file responsible for the user interface and model predictions.
- `cnn_lstm.h5`: The saved trained model file that contains the weights and architecture of the CNN-RNN model.
- `tokenizer.pickle`: The saved tokenizer that is used to preprocess user inputs consistently with the training data.

## Testing

Testing is essential to ensure the reliability of the application. To test the model:

1. **Input Diverse Emails**: Enter different examples of emails in the text area, including known spam and ham samples.
2. **Check Predictions**: Observe whether the predictions align with expectations. The model's accuracy can be improved by iteratively refining the training data and model architecture.

## About the Author

**Name**: Lawal Abdulazeez Faruq  
**Email**: [lawalabdulazeezfaruq@gmail.com](mailto:lawalabdulazeezfaruq@gmail.com)  
**LinkedIn**: [BnBazz Linkedin](https://www.linkedin.com/in/lawal-abdulazeezfaruq-0b8bb5232?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)  
**GitHub**: [BnBazz GitHub](https://github.com/Lawal0abdulazeez)

I am a data enthusiast passionate about machine learning and natural language processing. This project reflects my interest in developing practical applications that leverage technology for real-world problems.

## License

This project is licensed under the MIT License.
