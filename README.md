# Automatic Dialogue Act Classification

This project implements an application for classifying sentences (or parts of sentences) from comic book dialogues into categories based on their content, such as commands, questions, responses, etc.
The goal is to identify dialogue acts that define the function of a sentence within a conversation.

## Features

The application offers the following functionalities:
- **Model Training**: Train a classifier using a specified training dataset with different algorithms for feature extraction and classification.
- **Sentence Classification**: Use the GUI to classify new sentences based on a pre-trained model.
- **Accuracy Evaluation**: Evaluate the classification performance using accuracy as a metric.

## Methods Used

### Feature Extraction
- **Bag of Words (BoW)**: Represents text as a vector of word frequencies.
- **Term Frequency (TF)**: Normalized word frequency representation.
- **Term Frequency-Inverse Document Frequency (TF-IDF)**: Combines TF with inverse document frequency to weigh words based on their importance.

### Classification Algorithms
- **Naive Bayes Classifier**: A probabilistic model based on Bayes' theorem.
- **K-Nearest Neighbors (KNN)**: Classifies sentences based on the closest data points in the feature space.

### Additional Features
- **MLP Neural Network**: Implemented using Keras and TensorFlow for classification.
- **Clustering with K-Means**: Grouping similar sentences using K-Means clustering.
- **Comparison with Libraries**: Evaluate results using library implementations like scikit-learn.
- **English Document Classification**: Supports English sentence classification after preprocessing.

## Project Structure

The project consists of the following files:
- `main.py`: Main script to run the application.
- `scikitlearn.py`: Implementation using the scikit-learn library.
- `kmeans.py`: K-Means clustering implementation.
- `mlp.py`: Multi-Layer Perceptron (MLP) neural network implementation.
- `conllparser.py`: Parser for `.conll` data format.

## Requirements

Ensure you have the following installed:
- Python 3.x
- Libraries: `numpy`, `scikit-learn`, `tensorflow`, `keras`, `tkinter`

## Usage

### Training a Model

Run the following command to train a classifier:

```bash
python main.py <classifier_name> <class_file> <train_data> <test_data> <feature_algorithm> <classification_algorithm> <model_name>
```

### Running the GUI

To classify sentences using a pre-trained model, run:

```bash
python main.py <classifier_name> <model_name>
```

#### Example

```bash
python main.py NaiveBayes classes.txt train.txt test.txt TF-IDF NaiveBayes model.pickle
```

### Classifier Evaluation

Classifier performance is evaluated using the accuracy metric.
The results can be visualized using a confusion matrix for further analysis.

## Additional Features

1. Clustering: K-Means clustering is implemented with options to choose between Euclidean distance and cosine similarity.
2. MLP Neural Network: Configure the number of neurons, epochs, and learning rate for the model.
3. English Document Classification: Supports classification of English text after preprocessing.
