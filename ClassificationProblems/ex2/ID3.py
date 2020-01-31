"""
Decision Tree ID3 implementation
Name: Noa Madmon
ID: 323005785
"""

import math
import DB_util
from DB_util import DB


CLASSES = DB.classes


class DecisionNode:
    """
    class that represents every node in the decision tree
    """

    def __init__(self, attribute):
        self.attribute = attribute
        self.sub_trees = dict()

    def add_subtree(self, value, subtree):
        self.sub_trees[value] = subtree

    def __str__(self):
        return str(self.attribute)


class LeafNode(DecisionNode):
    """
    class that represents the leaves nodes in the decision tree.
    derives from DecisionNode
    """

    def __init__(self, classification):
        DecisionNode.__init__(self, DB.classes)
        if classification in self.attribute:
            self.classification = classification
        else:
            self.classification = DB.DEFAULT

    def __str__(self):
        return self.classification


def plurality_value(examples, classes):
    """
    returns the most common class in the given list of examples.
    """
    classification = DB_util.most_common(examples, classes)[0]
    return LeafNode(classification)


def information_gain(examples, attribute, db):
    """
    function that computes the information gain of the given attribute.
    high information gain means we get a less complex decision tree.
    """

    n = len(examples)
    count = DB_util.classification_counter(examples, db.classes)
    gain = entropy(count, n)

    for value in db.attr_domain[attribute]:
        exs = DB_util.choose(examples, attribute, value)
        m = len(exs)

        if m != 0:
            count = DB_util.classification_counter(exs, db.classes)
            gain = gain - (m/n) * entropy(count, m)

    return gain


def entropy(counter, n):
    """
    entropy calculation.
    for more information: https://amethix.com/entropy-in-machine-learning/
    """

    probabilities = [abs(p/n) for p in counter.values()]
    ent = [p * math.log2(p) for p in probabilities if p != 0]
    return -sum(ent)


def best_attribute(examples, attributes, db):
    """
    this function chooses the attribute with the best info-gain.
    """

    gains = []
    for attr in attributes:
        gains.append(information_gain(examples, attr, db))
    return attributes.pop(gains.index(max(gains)))


def dtl(examples, attributes, db, default_tree=LeafNode(None)):
    """
    builds a decision tree using ID3 algorithm.
    """

    if len(examples) == 0:
        return default_tree
    elif len(examples) == 1:
        return LeafNode(examples[0].classification)

    classifications = set()
    for example in examples:
        classifications.add(example.classification)
    if len(classifications) == 1:
        return LeafNode(list(classifications)[0])

    if len(attributes) == 0:
        return plurality_value(examples, db.classes)

    best = best_attribute(examples, attributes, db)

    tree = DecisionNode(best)
    for value in db.attr_domain[best]:
        elements = DB_util.choose(examples, best, value)
        sub_tree = dtl(elements, [attr for attr in attributes if attr != best],
                       db, default_tree=plurality_value(examples, db.classes))
        tree.sub_trees[value] = sub_tree

    return tree


def save(root, filename="tree.txt"):
    """
    this function saves the tree in a file.
    """

    with open(filename, "w") as output_file:
        counter = 0
        for child in sorted(root.sub_trees.keys()):
            if isinstance(child, LeafNode):
                output_file.write(str(root) + "=" + str(child) + ":" +
                                  str(root.sub_trees[child].classification) + "\n")
            else:
                output_file.write(str(root) + "=" + str(child) + "\n")
                write_node(root.sub_trees[child], counter + 1, output_file)


def write_node(node, tabs, output_file):
    for child in sorted(node.sub_trees.keys()):
        if isinstance(node.sub_trees[child], LeafNode):
            output_file.write("\t" * tabs + "|" + str(node) + "=" + str(child) + ":"
                              + str(node.sub_trees[child].classification) + "\n")
        else:
            output_file.write("\t" * tabs + "|" + str(node) + "=" + str(child) + "\n")
            write_node(node.sub_trees[child], tabs+1, output_file)


def predict(root, sample):
    """
    predicting the sample's classification
    """

    node = root
    while node and sample:

        if isinstance(node, LeafNode):
            return node.classification
        attr = node.attribute
        node = node.sub_trees[sample[attr]]

    return None     # we'll never reach here if the tree is well built.


def accuracy(train, test, db):
    """
    computes the level of accuracy of dtl algorithm on the given test set.
    """
    acc = 0

    root = dtl(train, list(db.attributes.copy()), db)

    for sample in test:
        prediction = predict(root, sample)
        if prediction == sample.classification:
            acc += 1

    return acc / len(test)


def main():
    """
    creates a decision tree and saves it in a file
    """

    db = DB()
    DB_util.shuffle_db(db, seed=10)

    root = dtl(db.examples, list(db.attributes.copy()), db)

    save(root, filename="tree.txt")


if __name__ == "__main__":
    main()
