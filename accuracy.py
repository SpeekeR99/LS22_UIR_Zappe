from tkinter import *
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix


def accuracy(actual_class, classified_class):
    """
    Calculates the accuracy of the model
    :param actual_class: expected classes
    :param classified_class: predicted classes
    :return: accuracy of the model in percentage
    """
    good = 0
    sum_ = 0

    for actual, classified in zip(classified_class, actual_class):
        if classified == actual:
            good += 1
        sum_ += 1

    return 100 * good / sum_


def test_accuracy(classifier, test_sentences, expected_classes):
    """
    Calculates the accuracy of the model
    :param classifier: classifier model to be tested
    :param test_sentences: test data
    :param expected_classes: expected classes
    """
    predicted = []
    for vector in test_sentences:
        predicted.append(classifier.classify(vector))

    print("Accuracy of model: " + str(accuracy(expected_classes, predicted)) + " %")
    plot_confusion_matrix(expected_classes, predicted)


def plot_confusion_matrix(y_test, y_pred):
    """
    Creates a window with a plot of the confusion matrix
    :param y_test: expected classes
    :param y_pred: predicted classes
    """
    mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 8))
    labels = sorted(set(y_test))
    for i in range(len(labels)):
        labels[i] = labels[i][:5]
    my_colors = [(0, 0, 0)]
    for i in range(80):
        my_colors.append((0, 0.2 + 0.01 * i, 0))
    sns.heatmap(mat.T, square=True, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cbar=False,
                cmap=my_colors)
    plt.xlabel("true labels")
    plt.ylabel("predicted label")
    plt.yticks(rotation=0)
    # plt.show()

    window = Tk()
    window.title("Confusion Matrix")
    window.resizable(width=False, height=False)
    canvas = FigureCanvasTkAgg(plt.gcf(), master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    window.mainloop()
