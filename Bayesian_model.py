import re
import numpy as np
import DataProcessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def TrainAndTest(fileName):
    X = []  #Tweet texts
    y = []  #Corresponding labels
    
    with open(fileName, encoding="utf-8") as file:
        file = file.readlines()
        for l in file:
            text = DataProcessing.GetText(l)
            text = DataProcessing.ReduceSentence(text)
            label = DataProcessing.GetLabel(l)
            # If statement is here to be able to modify I- labels quickly
            if label == "I-":
                X.append(text)
                y.append(label)
            else:
                X.append(text)
                y.append(label)

    f1_score_train = 0
    f1_score_test = 0
    f1_score_train_macro = 0
    f1_score_test_macro = 0

    pipeline = Pipeline([('vect', CountVectorizer()),
               ('clf', MultinomialNB()),
              ])

    number_of_experiments = 3
    for i in range(0, number_of_experiments):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i, stratify=y)
        #Train (create vocabulary)
        pipeline.fit(X_train, y_train)

        #Training data score
        y_train_pred = []
        for x in X_train:
            train_pred = pipeline.predict([x])
            y_train_pred.append(train_pred)
        f1_score_train_curr = f1_score(y_train, y_train_pred, labels=["I+", "I-", "U"], average=None)
        #f1_score_train_curr = f1_score(y_train, y_train_pred, pos_label="I+", average="binary")
        f1_score_train_macro_curr = f1_score(y_train, y_train_pred, labels=["I+", "I-", "U"], average="macro")
        f1_score_train += (f1_score_train_curr/number_of_experiments)
        f1_score_train_macro += (f1_score_train_macro_curr/number_of_experiments)
        confusion_matrix_train = confusion_matrix(y_train, y_train_pred, labels=["I+", "I-", "U"])
        #confusion_matrix_train = confusion_matrix(y_train, y_train_pred, labels=["I+", "U"])

        #Testing
        count = 0
        y_test_pred = []
        for idx,x in enumerate(X_test):
            test_pred = pipeline.predict([x])
            if test_pred != y_test[idx]:
                count += 1
                print(f"{x}, pred: {test_pred}, actual: {y_test[idx]}")
            y_test_pred.append(test_pred)
        print(f"{count}\n")
        f1_score_test_curr = f1_score(y_test, y_test_pred, labels=["I+", "I-", "U"], average=None)
        #f1_score_test_curr = f1_score(y_test, y_test_pred, pos_label="I+", average="binary")
        f1_score_test_macro_curr = f1_score(y_test, y_test_pred, labels=["I+", "I-", "U"], average="macro")
        f1_score_test += (f1_score_test_curr/number_of_experiments)
        f1_score_test_macro += (f1_score_test_macro_curr/number_of_experiments)
        confusion_matrix_test = confusion_matrix(y_test, y_test_pred, labels=["I+", "I-", "U"])
        #confusion_matrix_test = confusion_matrix(y_test, y_test_pred, labels=["I+", "U"])

        #Save confusion matrix
        df_cm = pd.DataFrame(confusion_matrix_train, index = ["Informative+", "Informative-", "Uninformative"],
                  columns = ["Informative+", "Informative-", "Uninformative"])
        #df_cm = pd.DataFrame(confusion_matrix_train, index = ["Informative+", "Uninformative"],
        #          columns = ["Informative+", "Uninformative"])
        plt.figure(figsize = (10,7))
        hm = sn.heatmap(df_cm, annot=True, fmt='g')
        fig = hm.get_figure()
        fig.savefig(f"./Data/Scikit_confusion_matrix_train_{i}.png")
        fig.clf()

        df_cm = pd.DataFrame(confusion_matrix_test, index = ["Informative+", "Informative-", "Uninformative"],
                    columns = ["Informative+", "Informative-", "Uninformative"])
        #df_cm = pd.DataFrame(confusion_matrix_test, index = ["Informative+", "Uninformative"],
        #            columns = ["Informative+", "Uninformative"])
        plt.figure(figsize = (10,7))
        hm = sn.heatmap(df_cm, annot=True, fmt='g')
        fig = hm.get_figure()
        fig.savefig(f"./Data/Scikit_confusion_matrix_test_{i}.png")
        fig.clf()

        endtest_file = fileName[:-14]+"endtest.txt"
        FinalTest(pipeline, endtest_file, i)

    print(f"Training mean f1 score: {f1_score_train}")
    print(f"Testing mean f1 score: {f1_score_test}")
    print(f"Training f1 macro: {f1_score_train_macro}")
    print(f"Testing f1 macro: {f1_score_test_macro}")

def FinalTest(pipeline, fileName, seed):
    X_test = []  #Tweet texts
    y_test = []  #Corresponding labels
    
    with open(fileName, encoding="utf-8") as file:
        file = file.readlines()
        for l in file:
            text = DataProcessing.GetText(l)
            text = DataProcessing.ReduceSentence(text)
            label = DataProcessing.GetLabel(l)
            # If statement is here to be able to modify I- labels quickly
            if label == "I-":
                X_test.append(text)
                y_test.append(label)
            else:
                X_test.append(text)
                y_test.append(label)

    f1_score_test = 0
    f1_score_test_macro = 0

    y_test_labels = list(map(lambda x: 0 if x == "I+" else 1 if x == "I-" else 2, y_test))
    sorted_concat_y_ylabel_x = sorted(zip(y_test_labels, y_test, X_test), key=lambda pair: pair[0])
    sorted_x_test = [x for _, _, x in sorted_concat_y_ylabel_x]
    sorted_y_test = [y for _, y, _ in sorted_concat_y_ylabel_x]
    sorted_y_test_labels = [ylabel for ylabel, _, _ in sorted_concat_y_ylabel_x]

    number_of_experiments = 1
    for i in range(0, number_of_experiments):
        #Testing
        count = 0
        y_test_pred = []
        for idx,x in enumerate(sorted_x_test):
            test_pred = pipeline.predict([x])
            if test_pred != sorted_y_test[idx]:
                count += 1
                print(f"endtest ___ {x}, pred: {test_pred}, actual: {sorted_y_test[idx]}")
            y_test_pred.append(test_pred)
        print(f"{count}\n")
        f1_score_test_curr = f1_score(sorted_y_test, y_test_pred, labels=["I+", "I-", "U"], average=None)
        f1_score_test_macro_curr = f1_score(sorted_y_test, y_test_pred, labels=["I+", "I-", "U"], average="macro")
        f1_score_test += (f1_score_test_curr/number_of_experiments)
        f1_score_test_macro += (f1_score_test_macro_curr/number_of_experiments)
        confusion_matrix_test = confusion_matrix(sorted_y_test, y_test_pred, labels=["I+", "I-", "U"])

        #Save confusion matrix
        df_cm = pd.DataFrame(confusion_matrix_test, index = ["Informative+", "Informative-", "Uninformative"],
                    columns = ["Informative+", "Informative-", "Uninformative"])
        plt.figure(figsize = (10,7))
        hm = sn.heatmap(df_cm, annot=True, fmt='g')
        fig = hm.get_figure()
        fig.savefig(f"./Data/Scikit_confusion_matrix_endtest_{seed}.png")
        fig.clf()

        data = pd.DataFrame()
        y_test_pred = list(map(lambda x: 0 if x == "I+" else 1 if x == "I-" else 2, y_test_pred))
        data['tweets'] = [i for i in range(100)]
        data['label'] = y_test_pred

        g = sn.FacetGrid(data)
        g = g.map(plt.scatter, "tweets", "label", edgecolor="w")
        plt.plot(data['tweets'], sorted_y_test_labels, color='r')
        plt.gcf().set_size_inches(20, 5)
        plt.savefig(f'./Data/sorted_predictions_{seed}.png', dpi=299)
        plt.clf()

    print(f"Endtest testing f1 score: {f1_score_test}")
    print(f"Endtest testing f1 macro: {f1_score_test_macro}")