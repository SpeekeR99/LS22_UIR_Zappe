import sys
import random

from my_classifiers import *
from my_features import *
from accuracy import *


class MyVector:
    """
    A class to represent a vector that has a label and a cluster
    """
    def __init__(self, vector, cluster, class_label=None):
        """
        Initialize a vector with a label and a cluster
        :param vector: vector
        :param cluster: cluster number (index of the center)
        :param class_label: corresponding class to a center
        """
        self.vector = vector
        self.cluster = cluster
        self.class_label = class_label


class KMeansClassifier(Classifier):
    """
    K-means classifier
    """
    def __init__(self, features, expected_classes, list_of_classes, features_algo, bag, words_idf, max_iter=10):
        """
        Initialize a K-means classifier
        :param features: vector of feature vectors
        :param expected_classes: expected classes of vectors
        :param list_of_classes: list of all classes
        :param features_algo: 1 for bag of words, 2 for tf, 3 for tf-idf
        :param bag: bag of words (for indexing)
        :param words_idf: idf of words (for tf-idf)
        :param max_iter: maximum number of iterations
        """
        self.vectors = []
        self.create_vectors(features)
        self.centers = []
        self.create_centers(expected_classes, list_of_classes)
        self.k = len(list_of_classes)
        self.max_iter = max_iter
        self.features_algo = features_algo
        self.bag = bag
        self.words_idf = words_idf
        self.train()

    def create_vectors(self, features):
        """
        Creates objects of MyVector class from features
        :param features: vector of feature vectors
        """
        for vector in features:
            self.vectors.append(MyVector(vector, -1))

    def create_centers(self, expected_classes, list_of_classes):
        """
        Initializes first centers smartly, so that each class has a center
        :param expected_classes: expected classes of vectors
        :param list_of_classes: list of all classes
        """
        for cluster, class_ in enumerate(list_of_classes):
            while True:
                index = random.randint(0, len(self.vectors) - 1)
                if class_ == expected_classes[index]:
                    self.centers.append(MyVector(self.vectors[index].vector, cluster, class_label=class_))
                    break

    def train(self):
        """
        Train the classifier (calls k-means algorithm)
        """
        self.centers = k_means(self.vectors, self.centers, self.k, max_iter=self.max_iter)

    def classify(self, vector, default=None):
        """
        Predict the class of a sentence
        :param vector: sentence already converted to a vector of features from the outside
        :param default: NOT USED HERE (KNN purposes)
        :return: predicted class of the sentence
        """
        distances = []
        for center in self.centers:
            distances.append(euclidean_distance(vector, center.vector))
        index = distances.index(min(distances))
        return self.centers[index].class_label


def k_means(features, centers, k, max_iter=10):
    """
    K-means algorithm
    :param features: vectors to be clustered
    :param centers: initial centers of the clusters
    :param k: number of clusters
    :param max_iter: maximum number of iterations
    :return: final state of the centers of the clusters
    """
    changed = 1
    iter = 0
    while changed != 0 and iter < max_iter:
        iter += 1
        changed = 0
        for vector in features:
            distances = []
            for center in centers:
                distances.append(euclidean_distance(vector.vector, center.vector))
            cluster = distances.index(min(distances))
            if vector.cluster != cluster:
                changed += 1
                vector.cluster = cluster
        for i in range(k):
            center = [0] * len(features[0].vector)
            n = 0
            for vector in features:
                if vector.cluster == i:
                    n += 1
                    for j in range(len(vector.vector)):
                        center[j] += vector.vector[j]
            if n != 0:
                for j in range(len(center)):
                    center[j] /= n
                centers[i] = MyVector(center, i, centers[i].class_label)
        print("K-means iteration: " + str(iter))
    return centers


def euclidean_distance(x, y):
    """
    Calculate the euclidean distance between two vectors
    :param x: vector 1
    :param y: vector 2
    :return: euclidean distance between the two vectors
    """
    result = 0
    for i in range(len(x)):
        result += (x[i] - y[i]) * (x[i] - y[i])
    return sqrt(result)


def cosine_similarity(x, y):
    """
    Calculate the cosine similarity between two vectors
    :param x: vector 1
    :param y: vector 2
    :return: cosine similarity between the two vectors
    """
    result = 0
    divider1 = 0
    divider2 = 0
    for i in range(len(x)):
        result += x[i] * y[i]
        divider1 += x[i] * x[i]
        divider2 += y[i] * y[i]
    divider = sqrt(divider1) * sqrt(divider2)
    if divider == 0:
        return 0
    return result / divider


def train(list_of_classes_filepath, training_data, test_data, features_algo, max_iter, model_name):
    """
    Trains the model and saves it to the file
    :param list_of_classes_filepath: filepath to a file containing the classes
    :param training_data: filepath to a file containing the training data
    :param test_data: filepath to a file containing the test data
    :param features_algo: 1 for bag of words, 2 for tf, 3 for tf-idf
    :param max_iter: maximum number of iterations for k-means
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

    list_of_classes = []
    with open(list_of_classes_filepath, "r") as fp:
        line = fp.readline()
        while line:
            list_of_classes.append(line.strip())
            line = fp.readline()

    if features_algo == 1:
        classifier = KMeansClassifier(sentences_bag_of_words, train_expected_classes, list_of_classes, features_algo, bag, words_idf, max_iter)
    elif features_algo == 2:
        classifier = KMeansClassifier(sentences_tf, train_expected_classes, list_of_classes, features_algo, bag, words_idf, max_iter)
    else:
        classifier = KMeansClassifier(tf_idf_, train_expected_classes, list_of_classes, features_algo, bag, words_idf, max_iter)

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

    text = Label(window, text='Classified as:', font=('helvetica', 10))
    canvas.create_window(400, 420, window=text)

    answer = Label(window, text="", font=('helvetica', 10, 'bold'))
    canvas.create_window(400, 460, window=answer)

    def classify():
        punctuation = ['...', '.', "--", "-", '?', '!', ","]
        sentence = entry.get().lower()
        for punct in punctuation:
            sentence = sentence.replace(punct, " " + punct)
        if classifier.features_algo == 1:
            vector = bow_feature(sentence, classifier.bag)
        elif classifier.features_algo == 2:
            vector = tf_feature(sentence, classifier.bag)
        else:
            vector = tf_idf_feature(sentence, classifier.bag, classifier.words_idf)
        result = classifier.classify(vector)
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
    print("Expected use case 1 (training model): kmeans.py <list_of_classes.txt> <training_data.txt> <test_data.txt> "
          "<features_algorithm> <max_iter> <model_name>")
    print("<list_of_classes.txt> - file with list of classes")
    print("<training_data.txt> - file with training data")
    print("<test_data.txt> - file with test data")
    print("<features_algorithm> - algorithm for features extraction (1, 2 or 3) (1 = bag of words, 2 = tf, "
          "3 = tf-idf)")
    print("<max_iter> - maximum number of iterations for K-means algorithm")
    print("<model_name> - name of the model (without extension)\n")
    print("Expected use case 2 (using model): kmeans.py <model_name>")
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
        train(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), "models/" + sys.argv[6] + ".pickle")


if __name__ == '__main__':
    main()

# data/DA_tridy.txt data/train.txt data/test.txt 1 10 kmeans1
# kmeans1
