import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DDSM_2_Folders import GeneralMultiDataModels

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

AUCNormal = 20
AUCBenign = 21
AUCMalignant = 22

Folder_Path = pd.read_csv("D:\DDSM\MultiDataCSV\Multiclass_DataFrame_DDSM_Data.csv")
Reverse = True
Multiclass = 3

#Parameters = [Folder_Path, Title, Label, Data, Reverse]

ParametersTrainingFE = [Folder_Path, "Best accuracy training FE", "Percentage", AccuracyLast, False, GeneralMultiDataModels, Multiclass]
ParametersTrainingLE = [Folder_Path, "Best accuracy training LE", "Percentage", AccuracyFirst, False, GeneralMultiDataModels, Multiclass]
ParametersTesting = [Folder_Path, "Best accuracy testing", "Percentage", AccuracyTest, False, GeneralMultiDataModels, Multiclass]

ParametersLossTraining = [Folder_Path, "Best loss training", "Percentage", LossTraining, True, GeneralMultiDataModels, Multiclass]
ParametersLossTesting = [Folder_Path, "Best loss testing", "Percentage", LossTesting, True, GeneralMultiDataModels, Multiclass]

ParametersPrecision = [Folder_Path, "Best precision", "Percentage", Precision, False, GeneralMultiDataModels, Multiclass]
ParametersRecall = [Folder_Path, "Best recall", "Percentage", Recall, False, GeneralMultiDataModels, Multiclass]
ParametersF1Score = [Folder_Path, "Best F1score", "Percentage", F1Score, False, GeneralMultiDataModels, Multiclass]

ParametersTimeTraining = [Folder_Path, "Best time training", "Seconds", Timetraining, True, GeneralMultiDataModels, Multiclass]
ParametersTimeTesting = [Folder_Path, "Best time testing", "Seconds", TimeTest, True, GeneralMultiDataModels, Multiclass]

ParametersAUCN = [Folder_Path, "Best AUC Normal", "Percentage", AUCNormal, False, GeneralMultiDataModels, Multiclass]
ParametersAUCB = [Folder_Path, "Best AUC Benign", "Percentage", AUCBenign, False, GeneralMultiDataModels, Multiclass]
ParametersAUCM = [Folder_Path, "Best AUC Malignant", "Percentage", AUCMalignant, False, GeneralMultiDataModels, Multiclass]

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

BarCharModels(ParametersAUCN)
BarCharModels(ParametersAUCB)
BarCharModels(ParametersAUCM)