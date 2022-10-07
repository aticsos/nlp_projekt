#!/usr/bin/env python
import math
import sys, os, argparse, json
import pickle
import datetime

"""
  "News Classifier" 
  -------------------------
  This is a small interface for document classification. 
  Implement your own Naive Bayes classifier by completing 
  the class 'NaiveBayesDocumentClassifier' below.

  To run the code, 

  1. place the files 'train.json' and 'test.json' in the current folder.

  2. train your model on 'train.json' by calling > python classifier.py --train 

  3. apply the model to 'test.json' by calling > python classifier.py --apply

"""


class NaiveBayesDocumentClassifier:

    def __init__(self, vocabulary, filepath, epsilon=0.0001):
        """ The classifier should store all its learned information
            in this 'model' object. Pick whatever form seems appropriate
            to you. Recommendation: use 'pickle' to store/load this model! """
        #epsilon enthält den wert der anstelle von null benutzt wird da amsonten die gesamt warscheinlichkeit null wird (null mal irgendwas ist null)
        self.epsilon = epsilon
        self.vocab = vocabulary

        try:
            self.model = pickle.load(open(filepath, 'rb'))
        except:
            self.model = dict()

    def train(self, features, labels):
        """
        trains a document classifier and stores all relevant
        information in 'self.model'.

        @type features: dict
        @param features: Each entry in 'features' represents a document
                         by its so-called bag-of-words vector. 
                         For each document in the dataset, 'features' contains 
                         all terms occurring in the document and their frequency
                         in the document:
                         {
                           'doc1.html':
                              {
                                'the' : 7,   # 'the' occurs seven times
                                'world': 3, 
                                ...
                              },
                           'doc2.html':
                              {
                                'community' : 2,
                                'college': 1, 
                                ...
                              },
                            ...
                         }
        @type labels: dict
        @param labels: 'labels' contains the class labels for all documents
                       in dictionary form:
                       {
                           'doc1.html': 'arts',       # doc1.html belongs to class 'arts'
                           'doc2.html': 'business',
                           'doc3.html': 'sports',
                           ...
                       }
        """
        #gesamt menge Der Dokumente
        gesamt = len(features.keys())
        #speichert die chancen das eine kategori drankommt
        categories = dict()
        #speichert die chancen ob eine wort in einem artikel vorhanden ist
        props = dict()

        #zählt wie oft eine artikel art vorkommt
        for label in labels.values():
            categories[label] = categories.get(label, 0) + 1

        #iteriert über alle artikel
        for item in labels.items():
            #hollt iteriert über alle wörter in dem artikel und zählt sie zusammen
            for vocs in features[item[0]]:
                try:
                    props[item[1]][vocs] = props[item[1]].get(vocs, 0) + 1
                except KeyError:
                    props[item[1]] = dict()
                    props[item[1]][vocs] = props[item[1]].get(vocs, 0) + 1

        for vocdic in props.items():
            # rechnet die chance aus das eine wort in einem artikel vorkommt
            for key in vocdic[1].keys():
                #wie oft das wort vorkommt geteilt dursch die anzahl an wörter insgesamt für dises label
                vocdic[1][key] /= categories[vocdic[0]]

        self.model["props"] = props

        print(self.model["props"])

        for key in categories.keys():
            #erechnet die grund chance ob eine artikel dran kommt
            categories[key] /= gesamt

        self.model["categories"] = categories

        pickle_out = open("news_classifier_model.pickle", "wb")
        pickle.dump(self.model, pickle_out)
        pickle_out.close()

    def apply(self, features):
        """
        applies a classifier to a set of documents. Requires the classifier
        to be trained (i.e., you need to call train() before you can call apply()).

        @type features: dict
        @param features: see above (documentation of train())

        @rtype: dict
        @return: For each document in 'features', apply() returns the estimated class.
                 The return value is a dictionary of the form:
                 {
                   'doc1.html': 'arts',
                   'doc2.html': 'travel',
                   'doc3.html': 'sports',
                   ...
                 }
        """
        #falls das program noch nicht traniert ist kommt ne error messag
        if not self.model:
            raise RuntimeError("Not trained")

        else:
            #dictonary in das die lösung gepackt wierd
            label = dict()
            #iteriert über alle artikel die geteste werden sollen
            for key in features.keys():
                # prediction wird mit -unendlich inizalisiert
                prediction = [-float('inf'), None]
                #iteriert über alle der KI bekanten labels
                for cat in self.model["props"].keys():
                    #füge  die grund warscheinlichkeit hinzu
                    certainty = math.log10(self.model["categories"][cat])
                    #iteriert über die bekanten wörter
                    for word in self.vocab:
                        #wen das aktuelle wort in dem zu überprüfenden artikel vorhanden ist
                        if word in features[key]:
                            #wen ja dan rechne die warscheinlichkeit das dises wort in diser art von artikel vorkommt auf die gesamt warscheinlichkeit drauf
                            #wir logaritmiren da die warscheinlichkeiten sonst zu kelein werden
                            certainty += math.log10(self.model["props"][cat].get(word, self.epsilon))
                        else:
                            #wen nein rechene die warscheinlichkeit das dass wort nicht in dem artikel ist hin zu
                            certainty += math.log10(1 - self.model["props"][cat].get(word, self.epsilon))
                    #falls die neu ausgerchnete warscheinlichkeit besser ist alls die in prediction gespeicherte dan ersetze sie
                    if certainty > prediction[0]:
                        prediction = certainty, cat
                #packe die lösung für den artikel in die label dictonary
                label[key] = prediction[1]
            #so ballt alle artikel bewertet wurde gib die liste zurück
            return label



if __name__ == "__main__":

    # parse command line arguments (no need to touch)
    parser = argparse.ArgumentParser(description='A document classifier.')
    parser.add_argument('--train', help="train the classifier", action='store_true')
    parser.add_argument('--apply', help="apply the classifier (you'll need to train or load" \
                                        "a trained model first)", action='store_true')
    parser.add_argument('--inspect', help="get some info about the learned model",
                        action='store_true')

    args = parser.parse_args()

    classifier = NaiveBayesDocumentClassifier(json.load(open("train_filtered.json"))['vocabulary'].keys(), "news_classifier_model.pickle", 0.0004)


    def read_json(path):
        with open(path) as f:
            data = json.load(f)['docs']
            features, labels = {}, {}
            for f in data:
                features[f] = data[f]['tokens']
                labels[f] = data[f]['label']
        return features, labels


    if args.train:
        features, labels = read_json('train_filtered.json')
        classifier.train(features, labels)

    if args.apply:
        features, labels = read_json('test_filtered.json')
        result = classifier.apply(features)

    errors = 0
    for key in result.keys():
        if not result[key] == labels[key]:
            errors += 1

    print(errors)

    error_rate = errors / len(labels)

    print(error_rate)
