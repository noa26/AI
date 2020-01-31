"""
This module handles the DB manipulation
It reads, shuffles, and handles the dataset

Name: Noa Madmon
ID: 323005785
"""

import csv
import random
import collections
from collections import Counter


Data = collections.namedtuple('Data', ['train', 'test', 'validate'])


class Example:
    """
    Example class
    used to save each sample at a simpler way.
    approached as a dictionary.
    """

    def __init__(self, attributes, values, db):
        self.features = dict()
        i = 0

        for i in range(len(attributes)):
            self.features[attributes[i]] = values[i]
            db.attr_domain[attributes[i]].add(values[i])
        self.classification = values[i+1]

    def __getitem__(self, key):
        return self.features[key]


class DB:
    """
    database handling class
    saves the data and performs operations on it.
    """

    DEFAULT = "yes"
    classes = ["yes", "no"]

    def __init__(self, file='dataset.txt', dlm='\t'):
        self.filename = file
        self.delimiter = dlm
        self.examples = []
        self.attributes = []
        self.attr_domain = dict()
        self.data = None
        self._read_file()

    def _read_file(self):
        with open(self.filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=self.delimiter)
            self.attributes = next(reader)[:-1]

            for attr in self.attributes:
                self.attr_domain[attr] = set()

            for row in reader:
                self.examples.append(Example(self.attributes, row, self))

    def __len__(self):
        return len(self.examples)

    def shuffle_data(self, sizes=(0, 0, 0)):
        exs = self.examples.copy()
        random.seed(10)
        random.shuffle(exs)
        if sizes[0] == 0:
            return Data(exs, [], [])
        else:
            train = exs[0:sizes[0]]
            test = exs[sizes[0]:(sizes[0] + sizes[1])]
            validate = exs[(sizes[0] + sizes[1]):]
            return Data(train, test, validate)


def choose(examples, attribute, value):
    """
    returns all of the examples where it's <attribute> value is value
    """
    exs = []
    for example in examples:
        if example[attribute] == value:
            exs.append(example)
    return exs


def most_common(examples, classes):
    """
    returns the most common classification of a given set of examples.
    return (class, number_of_occurrences)
    """
    count = classification_counter(examples, classes)
    return count.most_common(1)[0]


def classification_counter(examples, classes):
    """
    returns a counter that counts how many instances every class has in examples.
    """
    count = Counter([c for c in classes])
    for c in classes:
        count[c] = 0
    for example in examples:
        count[example.classification] += 1
    return count


def shuffle_db(db, seed=10):
    """
    shuffles the examples in the db
    """
    random.seed(seed)
    random.shuffle(db.examples)


def new_db(db):
    """
    just a function I created to see if I can handle different data sets with the same format
    """
    prefix = ""
    with open("trashDB.txt", "w") as out:
        for a in db.attributes[2:]:
            out.write(prefix + a)
            prefix = "\t"
        out.write("\n")
        for e in db.examples:
            prefix = ""
            for v in list(e.features.values())[2:]:
                out.write(prefix + v)
                prefix = "\t"
            out.write("\n")
    print("created")
