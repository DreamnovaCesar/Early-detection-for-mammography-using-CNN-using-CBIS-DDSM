import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DDSM_2_Folders import DataModelsML_FO
from DDSM_2_Folders import DataModelsML_SO

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

AUC = 13

Folder_Path = pd.read_csv("D:\DDSM\Calc Mammography\DataCSV\DataFrame_Binary_DDSM_Data_ML_SO.csv")
Biclass = 2

#Parameters = [Folder_Path, Title, Label, Data, Reverse]

ParametersAccuracy = [Folder_Path, "Best accuracy", "Percentage", Accuracy, False, DataModelsML_FO, Biclass]

ParametersPrecision = [Folder_Path, "Best precision", "Percentage", Precision, False, DataModelsML_FO, Biclass]
ParametersRecall = [Folder_Path, "Best recall", "Percentage", Recall, False, DataModelsML_FO, Biclass]
ParametersF1Score = [Folder_Path, "Best F1score", "Percentage", F1Score, True, DataModelsML_FO, Biclass]

ParametersTimetraining = [Folder_Path, "Best time training", "Seconds", Timetraining, True, DataModelsML_FO, Biclass]

ParametersAUC = [Folder_Path, "Best AUC", "Percentage", AUC, False, DataModelsML_FO, Biclass]

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

BarCharModels(ParametersAUC)

Folder_Path = pd.read_csv("D:\DDSM\Calc Mammography\DataCSV\DataFrame_Binary_DDSM_Data_ML_FO.csv")

ParametersAccuracy = [Folder_Path, "Best accuracy", "Percentage", Accuracy, False, DataModelsML_SO, Biclass]

ParametersPrecision = [Folder_Path, "Best precision", "Percentage", Precision, False, DataModelsML_SO, Biclass]
ParametersRecall = [Folder_Path, "Best recall", "Percentage", Recall, False, DataModelsML_SO, Biclass]
ParametersF1Score = [Folder_Path, "Best F1score", "Percentage", F1Score, True, DataModelsML_SO, Biclass]

ParametersTimetraining = [Folder_Path, "Best time training", "Seconds", Timetraining, True, DataModelsML_SO, Biclass]

ParametersAUC = [Folder_Path, "Best AUC", "Percentage", AUC, False, DataModelsML_SO, Biclass]

BarCharModels(ParametersAccuracy)

BarCharModels(ParametersPrecision)
BarCharModels(ParametersRecall)
BarCharModels(ParametersF1Score)

BarCharModels(ParametersTimetraining)

BarCharModels(ParametersAUC)