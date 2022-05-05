import sys

from my_classifiers import *
from my_features import *
from accuracy import *


def train(list_of_classes, training_data, test_data, features_algo, classifier_algo, model_name):
    """
    Trains the model and saves it to the file
    :param list_of_classes: filepath to a file containing the classes
    :param training_data: filepath to a file containing the training data
    :param test_data: filepath to a file containing the test data
    :param features_algo: 1 for bag of words, 2 for tf, 3 for tf-idf
    :param classifier_algo: 1 for Naive Bayes, 2 for KNN
    :param model_name: name of the future model to be saved as
    """
    train_sentences = read_sentences_from_file(training_data)
    test_sentences = read_sentences_from_file(test_data)

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

    train_expected_classes = []
    for sentence in train_sentences:
        train_expected_classes.append(sentence.expected_class_)

    if classifier_algo == 1:
        if features_algo == 1:
            classifier = NaiveBayes(sentences_bag_of_words, list_of_classes, train_expected_classes, bag, words_idf)
        elif features_algo == 2:
            classifier = NaiveBayes(sentences_tf, list_of_classes, train_expected_classes, bag, words_idf)
        else:
            classifier = NaiveBayes(tf_idf_, list_of_classes, train_expected_classes, bag, words_idf)
    else:
        if features_algo == 1:
            classifier = KNN(sentences_bag_of_words, train_expected_classes, bag, words_idf)
        elif features_algo == 2:
            classifier = KNN(sentences_tf, train_expected_classes, bag, words_idf)
        else:
            classifier = KNN(tf_idf_, train_expected_classes, bag, words_idf)

    save_classifier(classifier, model_name)

    test_vectors = []
    test_expected_classes = []
    for test_sentence in test_sentences:
        sentence = ""
        for word in test_sentence.words:
            sentence += word + " "
        vector = bow_feature(sentence, bag)
        test_vectors.append(vector)
        test_expected_classes.append(test_sentence.expected_class_)

    test_accuracy(classifier, test_vectors, test_expected_classes)


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

    if isinstance(classifier, KNN):
        k_label = Label(window, text="Enter k:", font=("helvetica", 10))
        canvas.create_window(650, 250, window=k_label)

        k_entry = Entry(window)
        k_entry.insert(0, "1")
        canvas.create_window(650, 280, window=k_entry, height=20, width=20)

    text = Label(window, text='Classified as:', font=('helvetica', 10))
    canvas.create_window(400, 420, window=text)

    answer = Label(window, text="", font=('helvetica', 10, 'bold'))
    canvas.create_window(400, 460, window=answer)

    def classify():
        punctuation = ['...', '.', "--", "-", '?', '!', ","]
        sentence = entry.get().lower()
        for punct in punctuation:
            sentence = sentence.replace(punct, " " + punct)
        vector = bow_feature(sentence, classifier.bag)
        if isinstance(classifier, KNN):
            k = int(k_entry.get())
            if k <= 0:
                k = 1
            if k % 2 == 0:
                k += 1
            k_entry.delete(0, END)
            k_entry.insert(0, str(k))
            answer.config(text=classifier.classify(vector, k))
        else:
            answer.config(text=classifier.classify(vector))

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
    print("Expected use case 1 (training model): main.py <list_of_classes.txt> <training_data.txt> <test_data.txt> "
          "<features_algorithm> <classifier_algorithm> <model_name>")
    print("<list_of_classes.txt> - file with list of classes")
    print("<training_data.txt> - file with training data")
    print("<test_data.txt> - file with test data")
    print("<features_algorithm> - algorithm for features extraction (1, 2 or 3) (1 = bag of words, 2 = tf, "
          "3 = tf-idf)")
    print("<classifier_algorithm> - algorithm for classifier (1 or 2) (1 = Naive Bayes, 2 = KNN)")
    print("<model_name> - name of the model (without extension)\n")
    print("Expected use case 2 (using model): main.py <model_name>")
    print("<model_name> - name of the model (without extension)")


def main():
    """
    Handles arguments from cmd line and calls the appropriate functions
    """
    number_of_args = len(sys.argv)
    if number_of_args != 2 and number_of_args != 7:
        print("\nInvalid number of parameters!\n")
        print_help()
        exit(1)
    if number_of_args == 2:  # loading model
        load("models/" + sys.argv[1] + ".pickle")
    if number_of_args == 7:  # training model
        train(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), "models/" + sys.argv[6] +
              ".pickle")


if __name__ == '__main__':
    main()

# data/DA_tridy.txt data/train.txt data/test.txt 1 1 bayes1
# bayes1
