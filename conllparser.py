import sys

from my_features import *


def read_conll(filename):
    """
    Reads the conll file and creates sentences and remembers their expected class
    :param filename: filepath to the conll file
    :return: list of Sentence objects
    """
    tmp = []
    with open(filename, 'r') as fp:
        sentence = Sentence("", [])
        line = fp.readline()
        while line:
            if line == "\n":
                tmp.append(sentence)
                sentence = Sentence("", [])
                line = fp.readline()
                continue
            line = line.split()
            sentence.expected_class_ = line.pop(len(line) - 1)
            sentence.words.append(line.pop(0))
            line = fp.readline()
    return tmp


def write_to_txt(sentences, filename):
    """
    Writes the sentences to the txt file in a format of expected class and then the sentence
    :param sentences: list of Sentence objects
    :param filename: filepath to the txt file
    """
    with open(filename, 'w') as fp:
        for sentence in sentences:
            fp.write(sentence.expected_class_ + " ")
            for word in sentence.words:
                fp.write(word + " ")
            fp.write("\n")


def create_list_of_classes(sentences):
    """
    Creates a set of classes from the sentences
    :param sentences: list of Sentence objects
    :return: set of classes
    """
    possible_classes = []
    for sentence in sentences:
        possible_classes.append(sentence.expected_class_)
    return set(possible_classes)


def convert(input_file, output_file, classes_file):
    """
    Converts the input file in conll format to the output file in txt format
    :param input_file: filepath to the input file in conll format
    :param output_file: filepath to the output file in txt format
    :param classes_file: if the user wants to create a file with the list of classes (optional)
    """
    input_conll = read_conll(input_file)
    write_to_txt(input_conll, output_file)
    if classes_file is not None:
        classes = create_list_of_classes(input_conll)
        with open(classes_file, "w") as fp:
            for c in classes:
                fp.write(c + "\n")


def print_help():
    """
    Prints the help menu if the user enters the wrong arguments
    """
    print("Expected use case 1 (training data): conllparser.py <input.conll> <output.txt> <classes.txt>")
    print("<input.conll> - path to input file in conll format")
    print("<output.txt> - path to output file in txt format")
    print("<classes.txt> - path to output file with list of classes\n")
    print("Expected use case 2 (test data): conllparser.py <input.conll> <output.txt>")
    print("<input.conll> - path to input file in conll format")
    print("<output.txt> - path to output file in txt format")


def main():
    """
    Handles arguments from cmd line and calls the appropriate functions
    """
    number_of_args = len(sys.argv)
    if number_of_args != 3 and number_of_args != 4:
        print("\nInvalid number of parameters!\n")
        print_help()
        exit(1)
    if number_of_args == 3:
        convert(sys.argv[1], sys.argv[2], None)
    if number_of_args == 4:
        convert(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == '__main__':
    main()

# data/train.conll data/train.txt data/classes.txt
# data/test.conll data/test.txt
