import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from itertools import cycle

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.multiclass import OneVsRestClassifier

#from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE

sm = SMOTE()

def DataModelsExtract(MainKeys, Arguments, Feature):

    DataFrame = pd.DataFrame()
  
    ClassSize = (len(Arguments[0]))
    Images = 1
    #Labels = 4

    if len(Arguments) == len(MainKeys):

        DicAruments = dict(zip(MainKeys, Arguments))

        for i in range(ClassSize):

            Dataset, X, Y, Name = Feature(list(DicAruments.values())[Images], i)
            DataFrame = pd.concat([DataFrame, Dataset], ignore_index = True, sort = False)
            Images += 2

    elif len(Arguments) > len(MainKeys):

            TotalArguments = len(Arguments) - len(MainKeys)

            for i in range(TotalArguments // 2):

                MainKeys.append('Images ' + str(i + 3))
                MainKeys.append('Labels ' + str(i + 3))

            DicAruments = dict(zip(MainKeys, Arguments))

            for i in range(ClassSize):

                Dataset, X, Y, Name = Feature(list(DicAruments.values())[Images], i)
                DataFrame = pd.concat([DataFrame, Dataset], ignore_index = True, sort = False)
                Images += 2

    elif len(Arguments) < len(MainKeys):

      raise ValueError('No se puede xD')

    def printDict(DicAruments):

      for i in range(2):
          print(list(DicAruments.items())[i])

    printDict(DicAruments)

    return DataFrame, Name

def MLConfigurationModels(DataFrame, Arguments, DataModels, DataCSV, Name):

    sc = StandardScaler()

    ClassSize = (len(Arguments[2]))

    if ClassSize == 2:
        ClassName = 'Biclass_'
    elif ClassSize >= 3:
        ClassName = 'Multiclass_'

    ColumnsDataframe = len(DataFrame.columns)

    X_Total = DataFrame.iloc[:, 0:ColumnsDataframe - 1].values
    Y_Total = DataFrame.iloc[:, -1].values

    print(X_Total)
    print(Y_Total)

    pd.set_option('display.max_rows', DataFrame.shape[0] + 1)
    #print(DataFrame)

    X_train, X_test, y_train, y_test = train_test_split(X_Total, Y_Total, test_size = 0.2, random_state = 1)

    X_train, y_train = sm.fit_resample(X_train, y_train)

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    print(y_train)
    print(y_test)

    Score = ModelsML(Arguments[0], Arguments[1], Arguments[2], X_train, y_train, X_test, y_test, DataModels)

    dst = ClassName + 'Dataframe_' + Name + '_' + Arguments[1] + '_' + Score[0] + '.csv'
    dstPath = os.path.join(DataCSV, dst)

    DataFrame.to_csv(dstPath)

    return Score

def updateCSV(Score, df, column_names, path, row):

    for i in range(len(Score)):
        df.loc[row, column_names[i]] = Score[i]
  
    df.to_csv(path, index = False)
  
    print(df)

def AllModelValues(df, column_names, path, Dataframe, ModelValues, DataModels, DataCSV, Name, StartData):

    for index, Model in enumerate(ModelValues):

        Score = MLConfigurationModels(Dataframe, Model, DataModels, DataCSV, Name)

        updateCSV(Score, df, column_names, path, index + StartData)

def ModelsML(ModelML, Technique, labels, X_train, y_train, X_test, y_test, Folder_Save):

    Height = 5
    Width = 12
    Annot_kws = 12
    font = 0.7
    h = 0.02

    Xsizefigure = 1
    Ysizefigure = 2

    digits = 4

    Score = []
    labels_Triclass_Num = []

    num_classes = len(labels)

    if num_classes == 2:
        LabelClassName = '_Biclass_'
    elif num_classes > 2:
        LabelClassName = '_Multiclass_'

    if len(labels) == 2:

        y_pred, Time_train, ModelName, classifier = ModelML(X_train, y_train, X_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        #cf_matrix = confusion_matrix(y_test, y_pred)

        print(cm)
        print(classification_report(y_test, y_pred, target_names = labels))

        # Accuracy
        Accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {round(Accuracy, digits)}")
        print("\n")

        # Precision
        Precision = precision_score(y_test, y_pred)
        print(f"Precision: {round(Precision, digits)}")

        print("\n")
        # Recall
        Recall = recall_score(y_test, y_pred)
        print(f"Recall: {round(Recall, digits)}")

        print("\n")
        # F1-score
        F1_Score = f1_score(y_test, y_pred)
        print(f"F1: {round(F1_Score, digits)}")

        print("\n")

        df_cm = pd.DataFrame(cm, range(len(cm)), range(len(cm[0])))

        plt.figure(figsize = (Width, Height))
        #technique
        plt.subplot(Xsizefigure, Ysizefigure, 1)
        sns.set(font_scale = font) # for label size

        ax = sns.heatmap(df_cm, annot = True, fmt = 'd') # font size
        #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        Auc = auc(fpr, tpr)

        plt.subplot(Xsizefigure, Ysizefigure, 2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label = ModelName + '(area = {:.4f})'.format(Auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc = 'lower right')

        dst = ModelName + LabelClassName + Technique + '.png'
        dstPath = os.path.join(Folder_Save, dst)

        plt.savefig(dstPath)

        #plt.show()

    elif len(labels) >= 3:

        y_pred, Time_train, ModelName, classifier = ModelML(X_train, y_train, X_test)

        for i in range(len(labels)):
            labels_Triclass_Num.append(i)
        
        print(y_pred)

        y_pred_roc = label_binarize(y_pred, classes = labels_Triclass_Num)
        y_test_roc = label_binarize(y_test, classes = labels_Triclass_Num)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        #cf_matrix = confusion_matrix(y_test, y_pred)

        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names = labels))

        # Accuracy
        Accuracy = accuracy_score(y_test, y_pred)
        print(f"Precision: {round(Accuracy, digits)}")

        # Precision
        Precision = precision_score(y_test, y_pred, average = 'weighted')
        print(f"Precision: {round(Precision, digits)}")

        print("\n")
        # Recall
        Recall = recall_score(y_test, y_pred, average = 'weighted')
        print(f"Recall: {round(Recall, digits)}")

        print("\n")
        # F1-score
        F1_Score = f1_score(y_test, y_pred, average = 'weighted')
        print(f"F1: {round(F1_Score, digits)}")

        print("\n")
        plt.figure(figsize = (Width, Height))

        df_cm = pd.DataFrame(cm, range(len(cm)), range(len(cm[0])))

        plt.subplot(Xsizefigure, Ysizefigure, 1)
        sns.set(font_scale = font) # for label size

        ax = sns.heatmap(df_cm, annot = True, fmt = 'd') # font size
        #ax.set_title('Seaborn Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_roc[:, i], y_pred_roc[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        colors = cycle(['blue', 'red', 'green', 'brown', 'purple', 'pink', 'orange', 'black', 'yellow', 'cyan'])
        
        plt.subplot(Xsizefigure, Ysizefigure, 2)
        plt.plot([0, 1], [0, 1], 'k--')

        for i, color, lbl in zip(range(num_classes), colors, labels):
            plt.plot(fpr[i], tpr[i], color = color, label = 'ROC Curve of class {0} (area = {1:0.4f})'.format(lbl, roc_auc[i]))

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc = 'lower right')

        dst = ModelName + LabelClassName + Technique + '.png'
        dstPath = os.path.join(Folder_Save, dst)

        plt.savefig(dstPath)

        #plt.show()
    
    Score.append(ModelName)
    Score.append(Accuracy)
    Score.append(Precision)
    Score.append(Recall)
    Score.append(F1_Score)
    Score.append(len(y_train))
    Score.append(len(y_test))
    Score.append(Time_train)
    Score.append(Technique)
    Score.append(cm[0][0])
    Score.append(cm[0][1])
    Score.append(cm[1][0])
    Score.append(cm[1][1])

    if num_classes == 2:
        Score.append(Auc)
    elif num_classes >= 3:
        for i in range(num_classes):
            Score.append(roc_auc[i])

    return Score
    
def SVM(X_train, y_train, X_test):

    ModelName = 'SVM'

    Begin_train = time.time()
    
    # Data Custom model
    classifier = SVC(kernel = 'rbf', C = 1)
    classifier.fit(X_train, y_train)

    End_train = time.time()

    Time_train = End_train - Begin_train
    
    y_pred = classifier.predict(X_test)

    return y_pred, Time_train, ModelName, classifier

def MultiSVM(X_train, y_train, X_test):

    ModelName = 'Multi SVM'

    Begin_train = time.time()
    
    # Data Custom model
    classifier = OneVsRestClassifier(SVC(kernel = 'rbf', C = 1))
    classifier.fit(X_train, y_train)

    End_train = time.time()

    Time_train = End_train - Begin_train
    
    y_pred = classifier.predict(X_test)

    return y_pred, Time_train, ModelName, classifier

def MLP(X_train, y_train, X_test):

    ModelName = 'MLP'

    Begin_train = time.time()
    
    # Data Custom model
    classifier = MLPClassifier(hidden_layer_sizes = [100] * 2, random_state = 1, max_iter = 2000)
    classifier.fit(X_train, y_train)

    End_train = time.time()

    Time_train = End_train - Begin_train
    
    y_pred = classifier.predict(X_test)

    return y_pred, Time_train, ModelName, classifier

def DT(X_train, y_train, X_test):

    ModelName = 'DT'

    Begin_train = time.time()
    
    # Data Custom model
    classifier = DecisionTreeClassifier(max_depth = 20)
    classifier.fit(X_train, y_train)

    End_train = time.time()

    Time_train = End_train - Begin_train
    
    y_pred = classifier.predict(X_test)

    return y_pred, Time_train, ModelName, classifier

def KNN(X_train, y_train, X_test):

    ModelName = 'KNN'

    Begin_train = time.time()
    
    # Data Custom model
    classifier = KNeighborsClassifier(n_neighbors = 7)
    classifier.fit(X_train, y_train)

    End_train = time.time()

    Time_train = End_train - Begin_train
    
    y_pred = classifier.predict(X_test)

    return y_pred, Time_train, ModelName, classifier

def RF(X_train, y_train, X_test):

    ModelName = 'RF'

    Begin_train = time.time()
    
    # Data Custom model
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    End_train = time.time()

    Time_train = End_train - Begin_train
    
    y_pred = classifier.predict(X_test)

    return y_pred, Time_train, ModelName, classifier

def GBC(X_train, y_train, X_test):

    ModelName = 'GBC'

    Begin_train = time.time()
    
    # Data Custom model
    classifier = GradientBoostingClassifier(n_estimators = 100, learning_rate = 1.0, max_depth = 2, random_state = 0)
    classifier.fit(X_train, y_train)

    End_train = time.time()

    Time_train = End_train - Begin_train
    
    y_pred = classifier.predict(X_test)

    return y_pred, Time_train, ModelName, classifier
