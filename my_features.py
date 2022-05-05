from math import log


class Sentence:
    """
    Sentence class for storing sentence and its expected class
    """
    def __init__(self, expected_class, words):
        """
        Initialize Sentence object
        :param expected_class: expected class of sentence
        :param words: words in sentence
        """
        self.expected_class_ = expected_class
        self.words = words


class Bag:
    """
    Bag class for storing words and their indexes for future vectors
    """
    def __init__(self):
        """
        Initialize Bag object
        """
        self.bag = {}
        self.i = 0

    def add(self, word):
        """
        Add word to bag if it is not in bag
        :param word: word to be added
        """
        if word not in self.bag:
            self.bag[word] = self.i
            self.i += 1


def read_sentences_from_file(filename):
    """
    Read sentences from file
    :param filename: filepath to file with sentences
    :return: array of Sentence objects
    """
    punctuation = ['...', '.', "--", "-", '?', '!', ","]
    sentences = []
    with open(filename, "r") as fp:
        line = fp.readline()
        while line:
            expected_class = line.split().pop(0)
            line = line.replace(expected_class, "")
            for punct in punctuation:
                line = line.replace(punct, " " + punct)
            words = line.split()
            sentences.append(Sentence(expected_class, words))
            line = fp.readline()
    return sentences


def bag_of_words(sentences, words_count=False):
    """
    Create bag of words for sentences
    :param sentences: sentences to create bag of words from
    :param words_count: True if you want to count words in bag, False otherwise
    :return: Bag object with words and their indexes for future vector creation and words count if words_count is True
    """
    bag = Bag()
    word_count = {}
    for sentence in sentences:
        for word in sentence.words:
            bag.add(word)
            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1
    if words_count:
        return bag, word_count
    return bag


def tf(sentences, bag):
    """
    Create tf vectors for sentences (Term Frequency)
    :param sentences: sentences to create tf vectors from
    :param bag: bag of words for indexing
    :return: tf vectors for sentences
    """
    result = []
    for sentence in sentences:
        vector = [0] * len(bag.bag)
        for word in sentence.words:
            if word in bag.bag:
                vector[bag.bag[word]] += 1 / len(sentence.words)
        result.append(vector)
    return result


def idf(sentences, bag):
    """
    Create idf vectors for sentences (Inverse Document Frequency)
    :param sentences: sentences to create idf vectors from
    :param bag: bag of words for indexing
    :return: idf vector for words
    """
    result = []
    for word in bag.bag:
        counter = 0
        for sentence in sentences:
            for word_ in sentence.words:
                if word == word_:
                    counter += 1
                    break
        result.append(log((1 + len(sentences)) / (1 + counter), 10))
    return result


def tf_idf(sentences_tf, words_idf):
    """
    Create tf-idf vectors for sentences (Term Frequency - Inverse Document Frequency)
    :param sentences_tf: tf vectors for sentences
    :param words_idf: idf vector for words
    :return: tf-idf vectors for sentences
    """
    result = []
    for sentence_tf in sentences_tf:
        vector = [0] * len(sentence_tf)
        for i in range(len(sentence_tf)):
            vector[i] = sentence_tf[i] * words_idf[i]
        result.append(vector)
    return result


def bow_feature(input_sentence, bag):
    """
    Create vector from sentence using Bag of Words feature
    :param input_sentence: sentence to create vector from
    :param bag: bag of words for indexing
    :return: vector for sentence
    """
    vector = [0] * len(bag.bag)
    for word in input_sentence.split():
        if word in bag.bag:
            vector[bag.bag[word]] += 1
    return vector


def tf_feature(input_sentence, bag):
    """
    Create vector from sentence using Term Frequency feature
    :param input_sentence: sentence to create vector from
    :param bag: bag of words for indexing
    :return: vector for sentence
    """
    vector = [0] * len(bag.bag)
    for word in input_sentence.split():
        if word in bag.bag:
            vector[bag.bag[word]] += 1 / len(input_sentence.split())
    return vector


def tf_idf_feature(input_sentence, bag, words_idf):
    """
    Create vector from sentence using Term Frequency - Inverse Document Frequency feature
    :param input_sentence: sentence to create vector from
    :param bag: bag of words for indexing
    :param words_idf: idf of words
    :return: vector for sentence
    """
    vector = tf_feature(input_sentence, bag)
    for i in range(len(vector)):
        vector[i] *= words_idf[i]
    return vector
