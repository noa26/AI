"""
K Nearest Neighbors implementation
Name: Noa Madmon
ID: 323005785
"""

import DB_util


def distribution(examples, db):
    """
    returns the distribution of the classes on the examples.
    """
    c = DB_util.classification_counter(examples, db.classes)
    n = len(examples)

    probabilities = dict()
    for classification in db.classes:
        probabilities[classification] = c[classification]/n

    return probabilities


def arg_max(likelihood):
    """
    returns the class with the maximum likelihood.
    """

    maximum, classification = None, None
    maximum = max(list(likelihood.values()))
    for classification in likelihood.keys():
        if likelihood[classification] == maximum:
            return classification
    return classification


def multiply(d1, d2):
    """
    multiplies two distributions and returns a new one.
    """

    if not d1:
        return d2
    elif not d2:
        return d1

    if list(d1.keys()) != list(d2.keys()):
        raise Exception("incompatible distributions at NaiveBayes.multiply\n" +
                        "d1: " + str(d1) + "d2: " + str(d2))

    d = dict()
    for classification in d1.keys():
        d[classification] = d2[classification] * d1[classification]
    return d


def naive_bayes(examples, sample, db):
    """
    implementation of the naive bayes algorithm.
    """

    classes_dist = distribution(examples, db)
    d = None

    for attr in db.attributes:
        exs = DB_util.choose(examples, attr, sample[attr])
        d = multiply(d, distribution(exs, db))
    likelihood = dict()
    for category in db.classes:
        likelihood[category] = d[category] / classes_dist[category]

    classification = arg_max(likelihood)
    return classification


def check_sample(examples, sample, db):
    nb_res = naive_bayes(examples, sample, db)

    if nb_res == sample.classification:
        return True
    else:
        return False


def accuracy(train, test, db):
    """
    calculates the accuracy of the algorithm on the test set.
    """

    acc = 0
    for sample in test:
        if check_sample(train, sample, db):
            acc += 1
    return acc / len(test)
