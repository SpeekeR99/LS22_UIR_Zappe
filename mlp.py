import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import save_model, load_model

from my_classifiers import *
from my_features import *
from accuracy import *


class MLPClassifier(Classifier):
    """
    Multi-layer perceptron classifier.
    """
    def __init__(self, train_x, train_y, classes, bag, words_idf, features_algo, hidden_units_1=128,
                 hidden_units_2=64, learning_rate=0.001, epochs=20, batch_size=16):
        """
        Initialize the classifier.
        :param train_x: Training data vectors
        :param train_y: Training data classes
        :param classes: Possible classes
        :param bag: Bag of words
        :param words_idf: IDF for each word
        :param features_algo: 1 - Bag of words, 2 - TF, 3 - TF-IDF
        :param hidden_units_1: Number of hidden units in the first hidden layer
        :param hidden_units_2: Number of hidden units in the second hidden layer
        :param learning_rate: Learning rate
        :param epochs: Number of epochs
        :param batch_size: Batch size
        """
        self.train_x = train_x
        self.train_y = train_y
        self.classes = classes
        self.inv_classes = {v: k for k, v in classes.items()}
        self.data_stuff = DataStuff(bag, words_idf, features_algo, self.inv_classes)
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.create_model()
        self.train()

    def create_model(self):
        """
        Creates the model
        """
        self.model = keras.Sequential(
            [
                keras.Input(shape=len(self.data_stuff.bag.bag)),
                layers.Dense(self.hidden_units_1, activation="relu"),
                layers.Dense(self.hidden_units_2, activation='relu'),
                layers.Dense(len(self.classes), activation="softmax"),
            ]
        )
        self.model.summary()

    def train(self):
        """
        Trains the model
        """
        self.model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=self.learning_rate),
                           metrics=["accuracy"])
        self.model.fit(self.train_x, self.train_y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.1)

    def classify(self, item, default=None):
        """
        Classifies an item
        :param item: sentence to be classified (already transformed into a vector from outside)
        :param default: NOT USED HERE (KNN purposes)
        :return: predicted class of the sentence as a string
        """
        return self.inv_classes[np.argmax(self.model.predict(np.array([item])))]


class DataStuff:
    """
    This class is used to store the data that is used in the MLPClassifier
    """
    def __init__(self, bag, words_idf, features_algo, inv_classes):
        self.bag = bag
        self.features_algo = features_algo
        self.words_idf = words_idf
        self.inv_classes = inv_classes


def train(classes_filepath, train_filepath, test_filepath, features_algo, hidden_units_1, hidden_units_2, epochs,
          batch_size, learning_rate, model_name):
    """
    Trains the model and saves it to the file
    :param classes_filepath: path to the file with classes
    :param train_filepath: filepath to a file containing the training data
    :param test_filepath: filepath to a file containing the test data
    :param features_algo: algorithm to use for features (1 - Bag of words, 2 - TF, 3 - TFIDF)
    :param hidden_units_1: number of hidden units in the first hidden layer
    :param hidden_units_2: number of hidden units in the second hidden layer
    :param epochs: number of epochs to train the model
    :param batch_size: batch size for training
    :param learning_rate: learning rate for training
    :param model_name: name of the future model to be saved as
    """
    train_sentences = read_sentences_from_file(train_filepath)
    test_sentences = read_sentences_from_file(test_filepath)

    bag = bag_of_words(train_sentences)
    sentences_bag_of_words = []
    for sentence in train_sentences:
        tmp = ""
        for word in sentence.words:
            tmp += word + " "
        sentences_bag_of_words.append(bow_feature(tmp, bag))
    sentences_tf = tf(train_sentences, bag)
    words_idf = idf(train_sentences, bag)
    tf_idf_ = tf_idf(sentences_tf, words_idf)

    if features_algo == 1:
        train_vectors = sentences_bag_of_words
    elif features_algo == 2:
        train_vectors = sentences_tf
    else:
        train_vectors = tf_idf_
    train_expected_classes = []
    for sentence in train_sentences:
        train_expected_classes.append(sentence.expected_class_)

    test_vectors = []
    test_expected_classes = []
    for test_sentence in test_sentences:
        sentence = ""
        for word in test_sentence.words:
            sentence += word + " "
        vector = bow_feature(sentence, bag)
        test_vectors.append(vector)
        test_expected_classes.append(test_sentence.expected_class_)

    train_x = train_vectors
    train_y = train_expected_classes
    test_x = test_vectors
    test_y = test_expected_classes

    classes = {}
    with open(classes_filepath, "r") as fp:
        line = fp.readline()
        i = 0
        while line:
            classes[line.strip()] = i
            line = fp.readline()
            i += 1

    for i in range(len(train_y)):
        train_y[i] = classes[train_y[i]]
    for i in range(len(test_y)):
        test_y[i] = classes[test_y[i]]

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    print("x train")
    print(train_x)
    print("y train")
    print(train_y)

    print(test_y)

    train_y = keras.utils.to_categorical(train_y, num_classes=len(classes))
    test_y = keras.utils.to_categorical(test_y, num_classes=len(classes))

    print(test_y)

    model = MLPClassifier(train_x, train_y, classes, bag, words_idf, features_algo,
                          hidden_units_1=hidden_units_1,
                          hidden_units_2=hidden_units_2, epochs=epochs, batch_size=batch_size,
                          learning_rate=learning_rate)

    save_model(model.model, model_name, overwrite=True)
    print("Model successfully saved to " + model_name)
    save_classifier(model.data_stuff, model_name + "_data.pickle")

    print("Accuracy of model: " + str(100 * model.model.evaluate(test_x, test_y, verbose=0)[1]) + " %")


def load(model_name):
    """
    Loads the model from the file and creates a GUI for it
    Lets the user enter a sentence and shows the predicted class
    :param model_name: name of the model to load
    """
    model = load_model(model_name)
    data_stuff = load_classifier(model_name + "_data.pickle")

    window = Tk()
    window.title(model_name)
    window.resizable(width=False, height=False)

    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = int((screen_width / 2) - 400)
    y = int((screen_height / 2) - 300)
    window.geometry("{}x{}+{}+{}".format(800, 600, x, y))

    canvas = Canvas(window, width=800, height=600)
    canvas.pack()

    heading = Label(window, text="Classify a sentence", font=("helvetica", 20, "bold"))
    canvas.create_window(400, 100, window=heading)

    entry = Entry(window)
    canvas.create_window(400, 280, window=entry, height=20, width=400)
    entry.focus_set()

    entry_label = Label(window, text="Write an input sentence:", font=("helvetica", 10))
    canvas.create_window(400, 250, window=entry_label)

    text = Label(window, text='Classified as:', font=('helvetica', 10))
    canvas.create_window(400, 420, window=text)

    answer = Label(window, text="", font=('helvetica', 10, 'bold'))
    canvas.create_window(400, 460, window=answer)

    def classify():
        punctuation = ['...', '.', "--", "-", '?', '!', ","]
        sentence = entry.get().lower()
        for punct in punctuation:
            sentence = sentence.replace(punct, " " + punct)
        if data_stuff.features_algo == 1:
            vector = bow_feature(sentence, data_stuff.bag)
        elif data_stuff.features_algo == 2:
            vector = tf_feature(sentence, data_stuff.bag)
        else:
            vector = tf_idf_feature(sentence, data_stuff.bag, data_stuff.words_idf)
        vector = np.array(vector)
        result = data_stuff.inv_classes[np.argmax(model.predict(np.array([vector])))]
        print(model.predict(np.array([vector])))
        print(np.argmax(model.predict(np.array([vector]))))
        print(data_stuff.inv_classes)
        print(result)
        answer.config(text=result)

    button = Button(text="Classify", command=classify, bg='green', fg='white', font=('helvetica', 10), )
    canvas.create_window(400, 360, window=button)

    def enter(event):
        if event.keysym == 'Return':
            classify()

    window.bind('<Return>', enter)

    window.mainloop()


def print_help():
    """
    Prints the help menu if the user enters the wrong arguments
    """
    print("Expected use case 1 (training model): mlp.py <list_of_classes.txt> <training_data.txt> <test_data.txt> "
          "<features_algorithm> <hidden_units_1> <hidden_units_2> <number_of_epochs> <batch_size> <learning_rate> "
          "<model_name>")
    print("<list_of_classes.txt> - file with list of classes")
    print("<training_data.txt> - file with training data")
    print("<test_data.txt> - file with test data")
    print("<features_algorithm> - algorithm for features extraction (1, 2 or 3) (1 = bag of words, 2 = tf, "
          "3 = tf-idf)")
    print("<hidden_units_1> - number of hidden units in the first hidden layer")
    print("<hidden_units_2> - number of hidden units in the second hidden layer")
    print("<number_of_epochs> - number of epochs")
    print("<batch_size> - batch size")
    print("<learning_rate> - learning rate")
    print("<model_name> - name of the model (without extension)\n")
    print("Expected use case 2 (using model): mlp.py <model_name>")
    print("<model_name> - name of the model (without extension)")


def main():
    """
    Handles arguments from cmd line and calls the appropriate functions
    """
    number_of_args = len(sys.argv)
    if number_of_args != 2 and number_of_args != 11:
        print("\nInvalid number of parameters!\n")
        print_help()
        exit(1)
    if number_of_args == 2:  # loading model
        load("models/" + sys.argv[1])
    if number_of_args == 11:  # training model
        train(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
              int(sys.argv[7]), int(sys.argv[8]), float(sys.argv[9]), "models/" + sys.argv[10])


if __name__ == '__main__':
    main()

# data/DA_tridy.txt data/train.txt data/test.txt 1 128 64 50 16 0.0001 MLP
# MLP
