import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DDSM_2_Folders import GeneralDataModels

NameModel = 0
AccuracyFirst = 2  
AccuracyLast = 3
AccuracyTest = 4

LossTraining = 5
LossTesting = 6

Precision = 9
Recall = 10
F1Score = 11

Timetraining = 12
TimeTest = 13

TN = 15
FP = 16
FN = 17
TP = 18

AUC = 20

Folder_Path = pd.read_csv("D:\DDSM\DataCSV\Biclass_DataFrame_DDSM_Data.csv")
Reverse = True
Biclass = 2

#Parameters = [Folder_Path, Title, Label, Data, Reverse]

ParametersTrainingFE = [Folder_Path, "Best accuracy training FE", "Percentage", AccuracyLast, False, GeneralDataModels, Biclass]
ParametersTrainingLE = [Folder_Path, "Best accuracy training LE", "Percentage", AccuracyFirst, False, GeneralDataModels, Biclass]
ParametersTesting = [Folder_Path, "Best accuracy testing", "Percentage", AccuracyTest, False, GeneralDataModels, Biclass]

ParametersLossTraining = [Folder_Path, "Best loss training", "Percentage", LossTraining, True, GeneralDataModels, Biclass]
ParametersLossTesting = [Folder_Path, "Best loss testing", "Percentage", LossTesting, True, GeneralDataModels, Biclass]

ParametersPrecision = [Folder_Path, "Best precision", "Percentage", Precision, False, GeneralDataModels, Biclass]
ParametersRecall = [Folder_Path, "Best recall", "Percentage", Recall, False, GeneralDataModels, Biclass]
ParametersF1Score = [Folder_Path, "Best F1score", "Percentage", F1Score, False, GeneralDataModels, Biclass]

ParametersTimeTraining = [Folder_Path, "Best time training", "Seconds", Timetraining, True, GeneralDataModels, Biclass]
ParametersTimeTesting = [Folder_Path, "Best time testing", "Seconds", TimeTest, True, GeneralDataModels, Biclass]

ParametersAUC = [Folder_Path, "Best AUC", "Percentage", AUC, False, GeneralDataModels, Biclass]

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
        plt.text(0, 68,
            'Worse value: ' + str(value) + ' -------> ' + str(XSlowest[0]), fontweight = 'bold')

    for index, value in enumerate(YFastest):
        plt.text(0, 70,
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

BarCharModels(ParametersTrainingFE)
BarCharModels(ParametersTrainingLE)
BarCharModels(ParametersTesting)

BarCharModels(ParametersLossTraining)
BarCharModels(ParametersLossTesting)

BarCharModels(ParametersPrecision)
BarCharModels(ParametersRecall)
BarCharModels(ParametersF1Score)

BarCharModels(ParametersTimeTraining)
BarCharModels(ParametersTimeTesting)

BarCharModels(ParametersAUC)

