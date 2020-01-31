"""
Name: Noa Madmon
ID: 323005785
"""
import DB_util
import ID3
import NaiveBayes
import KNN


def k_folds(db, k=5):
    """
    returns k sets of train and test tuples from the db.
    """

    train_len = int(4 * len(db) / 5)
    folds = []
    for i in range(k):
        DB_util.shuffle_db(db, seed=10)
        folds.append((db.examples[:train_len], db.examples[train_len:]))
    return folds


def accuracy(db, k=5):
    """
    calculates the accuracy of the 3 algorithms on the test set.
    1. DTL - ID3
    2. K Nearest Neighbors
    3. Naive Bayes
    """

    train, test = None, None
    id3_accuracy, knn_accuracy, nb_accuracy = 0, 0, 0

    for (train, test) in k_folds(db, k=k):
        id3_accuracy += ID3.accuracy(train, test, db)
        knn_accuracy += KNN.accuracy(train, test, db)
        nb_accuracy += NaiveBayes.accuracy(train, test, db)
        print("--{0}, {1}, {2}--".format(id3_accuracy, knn_accuracy, nb_accuracy))

    t = (int(100 * (id3_accuracy / 5)) / 100.0,
         int(100 * (knn_accuracy / 5)) / 100.0,
         int(100 * (nb_accuracy / 5)) / 100.0)

    return t


def get_data():
    train_db = DB_util.DB(file="train.txt")
    test_db = DB_util.DB(file="test.txt")
    return train_db.examples, test_db.examples, train_db


def main():
    """
    outputs the algorithms accuracy level.
    """

    train, test, db = get_data()
    tree = ID3.dtl(train, list(db.attributes.copy()), db)
    ID3.save(tree, filename="output.txt")

    id3_accuracy = int(100 * ID3.accuracy(train, test, db)) / 100.0
    knn_accuracy = int(100 * KNN.accuracy(train, test, db)) / 100.0
    nb_accuracy = int(100 * NaiveBayes.accuracy(train, test, db)) / 100.0

    with open("output.txt", "a") as out_file:
        out_file.write("\n{0}\t{1}\t{2}".format(id3_accuracy, knn_accuracy, nb_accuracy))
    print("done")


if __name__ == "__main__":
    main()
