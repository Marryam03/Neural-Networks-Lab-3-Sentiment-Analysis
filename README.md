# Sentiment Analysis with Deep Learning

## Overview

This project aims to build a sentiment analysis model using a combination of convolutional and recurrent neural networks. It preprocesses text data, utilizes GloVe embeddings for word representation, and trains a deep learning model to classify text sentiment.

## Dataset

It contains 1,600,000 tweets labelled as positive or negative.

## Requirements

To run this project, you will need the following libraries:

- Python 3.x
- NumPy
- Pandas
- TensorFlow
- Matplotlib
- NLTK
- Scikit-learn
- WordCloud

## Preprocessing

The preprocessing steps include:

1. Loading the dataset.
2. Dropping unnecessary columns.
3. Converting sentiment labels from numeric to textual form.
4. Cleaning text data by removing URLs, mentions, and non-alphanumeric characters.
5. Tokenizing and stemming the text.
6. Visualizing word clouds for positive and negative sentiments.

## Model

The model consists of the following layers:

1. **Embedding Layer**: Initialized with pre-trained GloVe embeddings.
2. **SpatialDropout1D**: For regularization.
3. **Conv1D**: Convolutional layer to capture local dependencies.
4. **Bidirectional LSTM**: For capturing sequential dependencies.
5. **Dense Layers**: Fully connected layers for learning complex representations.
6. **Output Layer**: Sigmoid activation for binary classification.

## Training

The model is trained using the following parameters:

- Learning Rate: 0.001
- Batch Size: 1024
- Epochs: 10
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Callbacks: ReduceLROnPlateau

The training process involves splitting the dataset into training and testing sets, tokenizing the text, and padding sequences.

## Evaluation

The model is evaluated using:

- Confusion Matrix
- Classification Report
- Accuracy, Precision, Recall, and F1-Score

## Results

The model achieves an accuracy of approximately 78% on the test set. The detailed classification report and confusion matrix are included in the notebook.
