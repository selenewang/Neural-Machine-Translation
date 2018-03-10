import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_dataset(language, dataset_path):
    with open('{}/{}.txt'.format(dataset_path, language), 'r', encoding='utf-8') as data:
        sentences = [sentence.rstrip('\n').split(' ') for sentence in data]
        sentences = [list(map(int, sent)) for sent in sentences]
    return sentences

def process_train_test_datasets(language_in='en', language_out='vi', dataset_path='data'):
    processed_dataset = dict()
    sentences_in = load_dataset(language_in, dataset_path)
    sentences_out = load_dataset(language_out, dataset_path)
    train_in, test_in, train_out, test_out = train_test_split(sentences_in, sentences_out, test_size=0.3, random_state=42)
    processed_dataset['train_{}'.format(language_in)]=train_in
    processed_dataset['test_{}'.format(language_in)]=test_in
    processed_dataset['train_{}'.format(language_out)]=train_out
    processed_dataset['test_{}'.format(language_out)]=test_out
    return processed_dataset

def showPlot(points, points2):
    plt.plot(points)
    plt.plot(points2)
    plt.show()