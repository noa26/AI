"""
K Nearest Neighbors implementation
Name: Noa Madmon
ID: 323005785
"""

import DB_util
from collections import namedtuple
from collections import Counter

Element = namedtuple('Element', ('sample', 'dist'))


class LimitedQueue:
    """
    A 'priority queue' where the smaller the distance,
    the higher the priority.
    """

    def __init__(self, k=5):
        self.elements = []
        self.limit = k
        self.max_elem = Element(None, 0)

    def add(self, elem, dist):
        if len(self.elements) < self.limit:
            self.elements.append(Element(elem, dist))
        elif dist < self.max_elem.dist:
            self.elements.remove(self.max_elem)
            self.elements.append(Element(elem, dist))

        self._update_min_max()

    def _update_min_max(self):
        maximum = Element(None, 0)
        for e in self.elements:
            if e.dist > maximum.dist:
                maximum = e
        self.max_elem = maximum


def hamming_dist(example1, example2):
    """
    calculates the hamming distance of 2 different examples.
    """
    dist = 0
    dist += len([1 for attr in example1.features.keys() if example1[attr] != example2[attr]])
    for attr in example1.features.keys():
        if example1[attr] != example2[attr]:
            dist += 1
    return dist


def knn(examples, sample, k=5):
    """
    implementation of the K Nearest Neighbors algorithm on the given examples.
    """

    q = LimitedQueue(k=k)
    for example in examples:
        q.add(example, hamming_dist(example, sample))

    classifications = Counter(["yes", "no"])
    for element in q.elements:
        classifications += Counter([element.sample.classification])
    common = classifications.most_common(1)[0]

    return common[0]


def check_sample(examples, sample):
    knn_res = knn(examples, sample)

    if knn_res == sample.classification:
        return True
    else:
        return False


def accuracy(train, test, db):
    """
    calculates the accuracy of the algorithm on the test set.
    """

    acc = 0
    for sample in test:
        if check_sample(train, sample):
            acc += 1
    return acc / len(test)
