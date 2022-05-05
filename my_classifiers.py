from math import sqrt, log, exp
import pickle


class Classifier:
    """
    Classifier is an abstract class
    """

    def train(self):
        """
        Train the classifier
        """
        pass

    def classify(self, item, default=None):
        """
        Predict the class of a sentence
        :param item: sentence to be classified
        :param default: KNN purposes (normally classifier doesn't need anything else, KNN needs k)
        :return: Predicted class of the sentence as a string
        """
        pass


class NaiveBayes(Classifier):
    """
    Naive Bayes classifier
    """

    def __init__(self, features, filepath_classes, expected_classes, bag, words_idf):
        """
        Initialize the classifier
        :param features: vector of feature vectors
        :param filepath_classes: filepath to the classes file
        :param expected_classes: list of expected classes
        :param bag: bag class (contains all the words in the training set)
        :param words_idf: idf of all the words in the training set
        """
        self.features = features
        self.filepath_classes = filepath_classes
        self.expected_classes = expected_classes
        self.bag = bag
        self.words_idf = words_idf
        self.classes = {}
        self.p_h = {}
        self.p_eh = {}
        self.train()

    def train(self):
        """
        Train the classifier
        """
        with open(self.filepath_classes, "r") as fp:
            line = fp.readline().strip()
            while line:
                self.classes[line] = 0
                line = fp.readline().strip()
        for expected_class in self.expected_classes:
            self.classes[expected_class] += 1
        for class_ in self.classes:
            self.p_h[class_] = (self.classes[class_] / len(self.expected_classes))
        for class_ in self.classes:
            sum_of_all = 0
            self.p_eh[class_] = [0] * len(self.features[0])
            for i, vector in enumerate(self.features):
                if self.expected_classes[i] != class_:
                    continue
                sum_of_all += sum(vector)
                for j in range(len(vector)):
                    self.p_eh[class_][j] += vector[j]
            self.p_eh[class_] = [(x + 1) / (sum_of_all + len(self.bag.bag)) for x in self.p_eh[class_]]

    def classify(self, input_vector, default=None):
        """
        Predict the class of a sentence
        :param input_vector: sentence to be classified (already transformed into a vector from outside)
        :param default: NOT USED HERE (KNN purposes)
        :return: predicted class of the sentence as a string
        """
        probabilities = {}
        for class_ in self.classes:
            # if self.p_h[class_] != 0:
            #     probabilities[class_] = log(self.p_h[class_])
            # else:
            probabilities[class_] = 0
            for i in range(len(input_vector)):
                if input_vector[i] != 0:
                    probabilities[class_] += log(self.p_eh[class_][i])
            # if probabilities[class_] == log(self.p_h[class_]):
            #     probabilities[class_] = 0

        max_prob = 0
        classified_class = "NOT_CLASSIFIABLE"
        for class_ in probabilities:
            if probabilities[class_] == 0:
                continue
            if exp(probabilities[class_]) > max_prob:
                max_prob = exp(probabilities[class_])
                classified_class = class_
        return classified_class


class KNN(Classifier):
    """
    KNN classifier
    """

    def __init__(self, features, expected_classes, bag, words_idf):
        """
        Initialize the classifier
        :param features: vector of feature vectors
        :param expected_classes: list of expected classes
        :param bag: bag class (contains all the words in the training set)
        :param words_idf: idf of all the words in the training set
        """
        self.features = features
        self.expected_classes = expected_classes
        self.bag = bag
        self.words_idf = words_idf

    def train(self):
        """
        Training is not used, since this classifier is straight up brute force nearest neighbor matching
        """
        pass

    def classify(self, input_vector, k=3):
        """
        Predict the class of a sentence
        :param input_vector: sentence to be classified (already transformed into a vector from outside)
        :param k: number of nearest neighbors to consider
        :return: predicted class of the sentence as a string
        """
        distances = []
        for vector in self.features:
            distances.append(self.euclidean_distance(input_vector, vector))
        classes = []
        already_done = []
        for i in range(k):
            max_distance = 0
            max_index = -1
            for j, distance in enumerate(distances):
                if distance > max_distance and j not in already_done:
                    max_distance = distance
                    max_index = j
            already_done.append(max_index)
        for index in already_done:
            if index == -1:
                classes.append("NOT_CLASSIFIABLE")
            else:
                classes.append(self.expected_classes[index])
        frequency = 0
        result = classes[0]
        for class_ in classes:
            curr_frequency = classes.count(class_)
            if curr_frequency > frequency:
                frequency = curr_frequency
                result = class_
        return result

    @staticmethod
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

    @staticmethod
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


def save_classifier(obj, filename):
    """
    Save the classifier to a file using pickle
    :param obj: classifier object to be saved
    :param filename: filepath to save the classifier to
    """
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    print("Model successfully saved to " + filename)


def load_classifier(filename):
    """
    Load a classifier from a file using pickle
    :param filename: filepath to load the classifier from
    :return: classifier object
    """
    with open(filename, 'rb') as inp:
        print("Model successfully loaded from " + filename)
        return pickle.load(inp)
