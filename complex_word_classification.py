"""Text classification for identifying complex words.

Author: Kristina Striegnitz and Anthony Piacentini

I affirm that I have carried out my academic endeavors with full
academic honesty. Anthony Piacentini

Complete this file for parts 2-4 of the project.

"""

from collections import defaultdict
import gzip
import numpy as np
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from syllables import count_syllables
from nltk.corpus import wordnet as wn

from evaluation import get_fscore, evaluate
nltk.download('movie_reviews')
nltk.download('wordnet')

def load_file(data_file):
    """Load in the words and labels from the given file."""
    words = []
    labels = []
    annotations = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline

def all_complex(data_file):
    """Label every word as complex. Evaluate performance on given data set. Print out
    evaluation results."""
    loaded = load_file(data_file)
    comp = []
    for i in range(len(loaded[0])):
        comp.append(1)
    evaluate(comp, loaded[1], 1)



### 2.2: Word length thresholding

def word_length_threshold(training_file, development_file):
    """Find the best length threshold by f-score and use this threshold to classify
    the training and development data. Print out evaluation results."""
    ## YOUR CODE HERE
    data = load_file(training_file)
    dev_data = load_file(development_file)
    classification_guess = []
    for i in range(len(data[0])):
        if(len(data[0][i]) >= 7):
           classification_guess.append(1)
        else:
            classification_guess.append(0)
    evaluate(classification_guess, data[1], 1)
            


### 2.3: Word frequency thresholding

def load_ngram_counts(ngram_counts_file):
    """Load Google NGram counts (i.e. frequency counts for words in a
    very large corpus). Return as a dictionary where the words are the
    keys and the counts are values.
    """
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    
    return counts

def word_frequency_threshold(training_file, development_file, counts):
    """Find the best frequency threshold by f-score and use this
    threshold to classify the training and development data. Print out
    evaluation results.
    """
    #min = 40
    #max = 47376829651
    data = load_file(training_file)
    dev_data = load_file(development_file)
    classification_guess = []
    for i in range(len(data[0])):
        if(data[0][i] in counts and counts[data[0][i]] >= 8100):
            classification_guess.append(1)
        else:
            classification_guess.append(0)
    evaluate(classification_guess, data[1], 1)

### 3.1: Naive Bayes

def naive_bayes(training_file, development_file, counts):
    """Train a Naive Bayes classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    load_train = load_file(training_file)
    load_dev = load_file(development_file)
    
    train_x = np.array([[len(word), counts[word]] for word in load_train[0]])
    train_y = np.array(load_train[1])

    # Standardize training data
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    train_x_scaled = (train_x - mean) / std

    # Load development data
    dev_x = np.array([[len(word), counts[word]] for word in load_dev[0]])
    dev_y = np.array(load_dev[1])

    # Standardize development data using train statistics
    dev_x_scaled = (dev_x - mean) / std

    # Train Logistic Regression classifier
    clf = GaussianNB()
    clf.fit(train_x_scaled, train_y)

    # Predict on development set
    Y_pred = clf.predict(dev_x_scaled)

    # Evaluate performance
    evaluate(Y_pred, dev_y, 1)


### 3.2: Logistic Regression

def logistic_regression(training_file, development_file, counts):
    """Train a Logistic Regression classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    load_train = load_file(training_file)
    load_dev = load_file(development_file)
    
    train_x = np.array([[len(word), counts[word]] for word in load_train[0]])
    train_y = np.array(load_train[1])

    # Standardize training data
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    train_x_scaled = (train_x - mean) / std

    # Load development data
    dev_x = np.array([[len(word), counts[word]] for word in load_dev[0]])
    dev_y = np.array(load_dev[1])

    # Standardize development data using train statistics
    dev_x_scaled = (dev_x - mean) / std

    # Train Logistic Regression classifier
    clf = LogisticRegression()
    clf.fit(train_x_scaled, train_y)

    # Predict on development set
    Y_pred = clf.predict(dev_x_scaled)

    # Evaluate performance
    evaluate(Y_pred, dev_y, 1)

def load_file_annotate(data_file):
    """Load in the words and labels from the given file."""
    words = []
    labels = []
    annotations = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
                annotations.append(int(line_split[2]))
            i += 1
    return words, labels, annotations
### 3.3: Build your own classifier

def my_classifier(training_file, development_file, counts):
    "Train your own classifier. Print out evaluation results on the training and development data."
    load_train = load_file(training_file)
    load_dev = load_file(development_file)
    
    train_x = np.array([[count_syllables(word), len(wn.synsets(word))] for word in load_train[0]])
    train_y = np.array(load_train[1])
    
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    train_x_scaled = (train_x - mean) / std
    
    dev_x = np.array([[count_syllables(word), len(wn.synsets(word))] for word in load_dev[0]])
    dev_y = np.array(load_dev[1])
    
    dev_x_scaled = (dev_x - mean) / std
    clf = SVC(kernel = "linear")
    clf.fit(train_x_scaled, train_y)
    
    Y_pred = clf.predict(dev_x_scaled)
    evaluate(Y_pred, dev_y, 1)
    



def baselines(training_file, development_file, counts):
    print("========== Baselines ===========\n")

    print("Majority class baseline")
    print("-----------------------")
    print("Performance on training data")
    all_complex(training_file)
    print("\nPerformance on development data")
    all_complex(development_file)

    print("\nWord length baseline")
    print("--------------------")
    word_length_threshold(training_file, development_file)

    print("\nWord frequency baseline")
    print("-------------------------")
    print("max ngram counts:", max(counts.values()))
    print("min ngram counts:", min(counts.values()))
    word_frequency_threshold(training_file, development_file, counts)

def classifiers(training_file, development_file, counts):
    print("\n========== Classifiers ===========\n")

    print("Naive Bayes")
    print("-----------")
    naive_bayes(training_file, development_file, counts)

    print("\nLogistic Regression")
    print("-----------")
    logistic_regression(training_file, development_file, counts)

    print("\nMy classifier")
    print("-----------")
    my_classifier(training_file, development_file, counts)

if __name__ == "__main__":
    training_file = "/Users/tonypiacentini/downloads/Project2/data/complex_words_training.txt"
    development_file = "/Users/tonypiacentini/downloads/Project2/data/complex_words_development.txt"
    test_file = "/Users/tonypiacentini/downloads/Project2/data/complex_words_test_unlabeled.txt"
    
    print("Loading ngram counts ...")
    ngram_counts_file = "/Users/tonypiacentini/downloads/Project2/data/ngram_counts.txt.gz"
    counts = load_ngram_counts(ngram_counts_file)
    

    
    #naive_bayes(training_file, development_file, counts)
    #logistic_regression(training_file, development_file, counts)
    #my_classifier(training_file, test_file, counts)
    baselines(training_file, development_file, counts)
    classifiers(training_file, development_file, counts)

    ## YOUR CODE HERE
    # Train your best classifier, predict labels for the test dataset and write
    # the predicted labels to the text file 'test_labels.txt', with ONE LABEL
    # PER LINE

