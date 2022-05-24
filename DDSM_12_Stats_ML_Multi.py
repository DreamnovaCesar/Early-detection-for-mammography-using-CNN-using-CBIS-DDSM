import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DDSM_2_Folders import MultiDataModelsML_FO
from DDSM_2_Folders import MultiDataModelsML_SO

NameModel = 0
Accuracy = 1  

Precision = 2
Recall = 3
F1Score = 4

Timetraining = 7

TN = 9
FP = 10
FN = 11
TP = 12

AUCN = 13
AUCB = 14
AUCM = 15

Folder_Path = pd.read_csv("D:\DDSM\Calc Mammography\MultiDataCSV\DataFrame_Multi_DDSM_Data_ML_FO.csv")
Multiclass = 3

#Parameters = [Folder_Path, Title, Label, Data, Reverse]

ParametersAccuracy = [Folder_Path, "Best accuracy", "Percentage", Accuracy, False, MultiDataModelsML_FO, Multiclass]

ParametersPrecision = [Folder_Path, "Best precision", "Percentage", Precision, False, MultiDataModelsML_FO, Multiclass]
ParametersRecall = [Folder_Path, "Best recall", "Percentage", Recall, False, MultiDataModelsML_FO, Multiclass]
ParametersF1Score = [Folder_Path, "Best F1score", "Percentage", F1Score, True, MultiDataModelsML_FO, Multiclass]

ParametersTimetraining = [Folder_Path, "Best time training", "Seconds", Timetraining, True, MultiDataModelsML_FO, Multiclass]

ParametersAUCN = [Folder_Path, "Best AUC Normal", "Percentage", AUCN, False, MultiDataModelsML_FO, Multiclass]
ParametersAUCB = [Folder_Path, "Best AUC Benign", "Percentage", AUCB, False, MultiDataModelsML_FO, Multiclass]
ParametersAUCM = [Folder_Path, "Best AUC Malignant", "Percentage", AUCM, False, MultiDataModelsML_FO, Multiclass]

def BarCharModels(Parameters):

    
    Path = Parameters[0]
    Title = Parameters[1]
    XLabel = Parameters[2]
    Data = Parameters[3]
    Reverse = Parameters[4]
    Folder_Save = Parameters[5]
    Num_classes = Parameters[6]

    if Num_classes == 2:
        LabelClassName = 'Biclass_'
    elif Num_classes > 2:
        LabelClassName = 'Multiclass_'

    YFast = []
    Yslow = []

    XFast = []
    Xslow = []

    XFastest = []
    YFastest = []

    XSlowest = []
    YSlowest = []

    # Initialize the lists for X and Y
    #data = pd.read_csv("D:\MIAS\MIAS VS\DataCSV\DataFrame_Binary_MIAS_Data.csv")
  
    df = pd.DataFrame(Path)
  
    X = list(df.iloc[:, 0])
    Y = list(df.iloc[:, Data])

    plt.figure(figsize = (10, 9))

    if Reverse == True:

        for index, (k, i) in enumerate(zip(X, Y)):
            if i < np.mean(Y):
                XFast.append(k)
                YFast.append(i)
            elif i >= np.mean(Y):
                Xslow.append(k)
                Yslow.append(i)

        for index, (k, i) in enumerate(zip(XFast, YFast)):
            if i == np.min(YFast):
                XFastest.append(k)
                YFastest.append(i)

        for index, (k, i) in enumerate(zip(Xslow, Yslow)):
            if i == np.max(Yslow):
                XSlowest.append(k)
                YSlowest.append(i)

    elif Reverse == False:

        for index, (k, i) in enumerate(zip(X, Y)):
            if i < np.mean(Y):
                Xslow.append(k)
                Yslow.append(i)
            elif i >= np.mean(Y):
                XFast.append(k)
                YFast.append(i)

        for index, (k, i) in enumerate(zip(XFast, YFast)):
            if i == np.max(YFast):
                XFastest.append(k)
                YFastest.append(i)

        for index, (k, i) in enumerate(zip(Xslow, Yslow)):
            if i == np.min(Yslow):
                XSlowest.append(k)
                YSlowest.append(i)

                
# Plot the data using bar() method
    plt.barh(XFast, YFast, label = "Better", color = 'lightcoral')
    plt.barh(XFastest, YFastest, label = "Best", color = 'red')
    plt.barh(Xslow, Yslow, label = "Bad", color = 'gray')
    plt.barh(XSlowest, YSlowest, label = "Worse", color = 'black')

    Postion = len(Y) - len(Yslow)

    for index, value in enumerate(YSlowest):
        plt.text(0, 40,
            'Worse value: ' + str(value) + ' -------> ' + str(XSlowest[0]), fontweight = 'bold')

    for index, value in enumerate(YFastest):
        plt.text(0, 41,
            'Best value: ' + str(value) + ' -------> ' + str(XFastest[0]), fontweight = 'bold')

    plt.legend()

    plt.title(Title)
    plt.xlabel(XLabel, fontsize = 8)
    plt.xticks(fontsize = 10)
    plt.ylabel("Models", fontsize = 8)
    plt.yticks(fontsize = 6)
    plt.grid(color = 'gray', linestyle = '-', linewidth = 0.2)

    dst = LabelClassName + Title + '.png'
    dstPath = os.path.join(Folder_Save, dst)

    plt.savefig(dstPath)

    # Show the plot
    #plt.show()

BarCharModels(ParametersAccuracy)

BarCharModels(ParametersPrecision)
BarCharModels(ParametersRecall)
BarCharModels(ParametersF1Score)

BarCharModels(ParametersTimetraining)

BarCharModels(ParametersAUCN)
BarCharModels(ParametersAUCB)
BarCharModels(ParametersAUCM)

Folder_Path = pd.read_csv("D:\DDSM\Calc Mammography\MultiDataCSV\DataFrame_Multi_DDSM_Data_ML_SO.csv")

ParametersAccuracy = [Folder_Path, "Best accuracy", "Percentage", Accuracy, False, MultiDataModelsML_SO, Multiclass]

ParametersPrecision = [Folder_Path, "Best precision", "Percentage", Precision, False, MultiDataModelsML_SO, Multiclass]
ParametersRecall = [Folder_Path, "Best recall", "Percentage", Recall, False, MultiDataModelsML_SO, Multiclass]
ParametersF1Score = [Folder_Path, "Best F1score", "Percentage", F1Score, True, MultiDataModelsML_SO, Multiclass]

ParametersTimetraining = [Folder_Path, "Best time training", "Seconds", Timetraining, True, MultiDataModelsML_SO, Multiclass]

ParametersAUCN = [Folder_Path, "Best AUC Normal", "Percentage", AUCN, False, MultiDataModelsML_SO, Multiclass]
ParametersAUCB = [Folder_Path, "Best AUC Benign", "Percentage", AUCB, False, MultiDataModelsML_SO, Multiclass]
ParametersAUCM = [Folder_Path, "Best AUC Malignant", "Percentage", AUCM, False, MultiDataModelsML_SO, Multiclass]

BarCharModels(ParametersAccuracy)

BarCharModels(ParametersPrecision)
BarCharModels(ParametersRecall)
BarCharModels(ParametersF1Score)

BarCharModels(ParametersTimetraining)

BarCharModels(ParametersAUCN)
BarCharModels(ParametersAUCB)
BarCharModels(ParametersAUCM)