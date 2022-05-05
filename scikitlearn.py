import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from my_classifiers import *
from my_features import *
from accuracy import *


class ScikitLearnClassifier(Classifier):
    """
    ScikitLearnClassifier is a wrapper for the scikit-learn classifier
    """
    def __init__(self, x_train, y_train):
        """
        Initializes the classifier (Multinomial Naive Bayes)
        :param x_train: training data
        :param y_train: test data
        """
        self.classifier = MultinomialNB()
        self.x_train = x_train
        self.y_train = y_train

    def train(self):
        """
        Trains the classifier (Calls the fit function)
        """
        self.classifier.fit(self.x_train, self.y_train)

    def classify(self, item, default=None):
        """
        Classifies the item (Calls the predict function)
        :param item: sentence to be classified (already transformed into a vector from outside)
        :param default: NOT USED HERE (KNN purposes)
        :return: predicted class of the sentence as a string
        """
        return self.classifier.predict(item)


def train(training_data, test_data, model_name):
    """
    Trains the model and saves it to the file
    :param training_data: filepath to a file containing the training data
    :param test_data: filepath to a file containing the test data
    :param model_name: name of the future model to be saved as
    """
    train_sentences = read_sentences_from_file(training_data)
    test_sentences = read_sentences_from_file(test_data)

    train_sentences_sentences = []
    train_sentences_classes = []
    for train_sentence in train_sentences:
        sentence = ""
        for word in train_sentence.words:
            sentence += word + " "
        train_sentences_classes.append(train_sentence.expected_class_)
        train_sentences_sentences.append(sentence)

    test_sentences_sentences = []
    test_sentences_classes = []
    for test_sentence in test_sentences:
        sentence = ""
        for word in test_sentence.words:
            sentence += word + " "
        test_sentences_classes.append(test_sentence.expected_class_)
        test_sentences_sentences.append(sentence)

    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(train_sentences_sentences)

    # tf_transformer = TfidfTransformer(use_idf=False).fit(x_train_counts)
    # x_train_tf = tf_transformer.transform(x_train_counts)

    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

    classifier = ScikitLearnClassifier(x_train_tfidf, train_sentences_classes)
    classifier.train()

    x_test_counts = count_vect.transform(test_sentences_sentences)
    x_test_tfidf = tfidf_transformer.transform(x_test_counts)

    y_pred = classifier.classify(x_test_tfidf)

    print("Accuracy of model (mine): " + str((accuracy(test_sentences_classes, y_pred))) + " %")
    print("Accuracy of model (scikit): " + str(100 * (accuracy_score(test_sentences_classes, y_pred))) + " %")

    classifier.count_vect = count_vect
    classifier.tfidf_transformer = tfidf_transformer

    save_classifier(classifier, model_name)

    plot_confusion_matrix(test_sentences_classes, y_pred)


def load(model_name):
    """
    Loads the model from the file and creates a GUI for it
    Lets the user enter a sentence and shows the predicted class
    :param model_name: name of the model to load
    """
    classifier = load_classifier(model_name)

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
        sentence = [sentence]
        sentence = classifier.count_vect.transform(sentence)
        sentence = classifier.tfidf_transformer.transform(sentence)
        sentence = classifier.classify(sentence)
        answer.config(text=sentence[0])

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
    print("Expected use case 1 (training model): scikitlearn.py <training_data.txt> <test_data.txt> <model_name>")
    print("<training_data.txt> - file with training data")
    print("<test_data.txt> - file with test data")
    print("<model_name> - name of the model (without extension)\n")
    print("Expected use case 2 (using model): scikitlearn.py <model_name>")
    print("<model_name> - name of the model (without extension)")


def main():
    """
    Handles arguments from cmd line and calls the appropriate functions
    """
    number_of_args = len(sys.argv)
    if number_of_args != 2 and number_of_args != 4:
        print("\nInvalid number of parameters!\n")
        print_help()
        exit(1)
    if number_of_args == 2:  # loading model
        load("models/" + sys.argv[1] + ".pickle")
    if number_of_args == 4:  # training model
        train(sys.argv[1], sys.argv[2], "models/" + sys.argv[3] + ".pickle")


if __name__ == '__main__':
    main()

# data/train.txt data/test.txt scikit1
# scikit1
