import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from DDSM_2_Folders import GeneralDataModels
from DDSM_2_Folders import GeneralDataModelsEsp

from DDSM_8_Preprocessing_9_Parameters import labels_Multiclass

from DDSM_8_Preprocessing_9_Parameters import RAWTechnique
from DDSM_8_Preprocessing_9_Parameters import NOTechnique
from DDSM_8_Preprocessing_9_Parameters import CLAHETechnique
from DDSM_8_Preprocessing_9_Parameters import HETechnique
from DDSM_8_Preprocessing_9_Parameters import UMTechnique
from DDSM_8_Preprocessing_9_Parameters import CSTechnique

from DDSM_8_Preprocessing_9_Parameters import Valid_split
from DDSM_8_Preprocessing_9_Parameters import Epochs

from DDSM_8_Preprocessing_9_Parameters import XsizeResized
from DDSM_8_Preprocessing_9_Parameters import YsizeResized

from XDDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_5 import Images_Normal
from XDDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_5 import Labels_Normal
from XDDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_5 import Images_BCalcification
from XDDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_5 import Labels_BCalcification
from XDDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_5 import Images_MCalcification
from XDDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_5 import Labels_MCalcification
from XDDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_5 import Images_BMass
from XDDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_5 import Labels_BMass
from XDDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_5 import Images_MMass
from XDDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_5 import Labels_MMass

from DDSM_7_1_CNN_Architectures import PreTrainedModels
from DDSM_7_1_CNN_Architectures import ResNet50_PreTrained
from DDSM_7_1_CNN_Architectures import ResNet50V2_PreTrained
from DDSM_7_1_CNN_Architectures import ResNet152_PreTrained
from DDSM_7_1_CNN_Architectures import ResNet152V2_PreTrained

from DDSM_7_1_CNN_Architectures import MobileNet_Pretrained
from DDSM_7_1_CNN_Architectures import MobileNetV3Small_Pretrained
from DDSM_7_1_CNN_Architectures import MobileNetV3Large_Pretrained

from DDSM_7_1_CNN_Architectures import Xception_Pretrained

from DDSM_7_1_CNN_Architectures import VGG16_PreTrained
from DDSM_7_1_CNN_Architectures import VGG19_PreTrained

from DDSM_7_1_CNN_Architectures import InceptionV3_PreTrained

from DDSM_7_1_CNN_Architectures import DenseNet121_PreTrained
from DDSM_7_1_CNN_Architectures import DenseNet201_PreTrained

from DDSM_7_1_CNN_Architectures import CustomCNNAlexNet12_Model

def ConfigurationModels(MainKeys, Arguments, MultiDataModels, MultiDataModelsEsp):

    TotalImage = []
    TotalLabel = []

    ClassSize = (len(Arguments[2]))
    Images = 7
    Labels = 8

    if len(Arguments) == len(MainKeys):
        
        DicAruments = dict(zip(MainKeys, Arguments))

        for i in range(ClassSize):

            for element in list(DicAruments.values())[Images]:
                TotalImage.append(element)
            
            #print('Total:', len(TotalImage))
        
            for element in list(DicAruments.values())[Labels]:
                TotalLabel.append(element)

            #print('Total:', len(TotalLabel))

            Images += 2
            Labels += 2

        #TotalImage = [*list(DicAruments.values())[Images], *list(DicAruments.values())[Images + 2]]
        
    elif len(Arguments) > len(MainKeys):

        TotalArguments = len(Arguments) - len(MainKeys)

        for i in range(TotalArguments // 2):

            MainKeys.append('Images ' + str(i + 3))
            MainKeys.append('Labels ' + str(i + 3))

        DicAruments = dict(zip(MainKeys, Arguments))

        for i in range(ClassSize):

            for element in list(DicAruments.values())[Images]:
                TotalImage.append(element)
            
            for element in list(DicAruments.values())[Labels]:
                TotalLabel.append(element)

            Images += 2
            Labels += 2

    elif len(Arguments) < len(MainKeys):

        raise ValueError('No se puede xD')

    #print(DicAruments)

    def printDict(DicAruments):

        for i in range(7):
            print(list(DicAruments.items())[i])

    printDict(DicAruments)

    print(len(TotalImage))
    print(len(TotalLabel))

    X_train, X_test, y_train, y_test = train_test_split(np.array(TotalImage), np.array(TotalLabel), test_size = 0.20, random_state = 3, shuffle = True)


    Arguments
    Score = PreTrainedModels(Arguments[0], Arguments[1], Arguments[2], Arguments[3], Arguments[4], ClassSize, Arguments[5], Arguments[6], X_train, y_train, X_test, y_test, MultiDataModels, MultiDataModelsEsp)
    #Score = PreTrainedModels(ModelPreTrained, technique, labels, Xsize, Ysize, num_classes, vali_split, epochs, X_train, y_train, X_test, y_test)
    return Score

def updateCSV(Score, df, column_names, path, row):

    print(df.head(len(df.index)))

    for i in range(len(Score)):
        df.loc[row, column_names[i]] = Score[i]
    
    print(df.head(len(df.index)))

    df.to_csv(path, index = False)
  
    print(df)

df = pd.read_csv("D:\DDSM\MultiDataCSV\DataFrame_Multi_DDSM_Data.csv")
path = "D:\DDSM\MultiDataCSV\DataFrame_Multi_DDSM_Data.csv"

MainKeys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs','Images 1', 'Labels 1', 'Images 2', 'Labels 2']
column_names = ["Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs", "Auc 1", "Auc 2", "Auc 3", "Auc 4", "Auc 5"]

ModelTest = ResNet50_PreTrained

ModelValues =   [ModelTest, NOTechnique, labels_Multiclass, XsizeResized, YsizeResized, Valid_split, Epochs, Images_Normal, Labels_Normal, Images_BCalcification, Labels_BCalcification, Images_MCalcification, Labels_MCalcification, Images_BMass, Labels_BMass, Images_MMass, Labels_MMass]

#ModelValues1 =  [ModelTest, NOTechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, NOImages_Calcification, NOLabels_Calcification, NOImages_Mass, NOLabels_Mass]

#ModelValues2 =  [ModelTest, CLAHETechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, CLAHEImages_Calcification, CLAHELabels_Calcification, CLAHEImages_Mass, CLAHELabels_Mass]
#ModelValues3 =  [ModelTest, HETechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, HEImages_BenignWC, HELabels_BenignWC, HEImages_Malignant, HELabels_Malignant]
#ModelValues4 =  [ModelTest, UMTechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, UMImages_BenignWC, UMLabels_BenignWC, UMImages_Malignant, UMLabels_Malignant]
#ModelValues5 =  [ModelTest, CSTechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, CSImages_BenignWC, CSLabels_BenignWC, CSImages_Malignant, CSLabels_Malignant]

#ModelValues = [MobileNetV3Small_Pretrained, CLAHETechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, Images_Normal, Labels_Normal, Images_Benign, Labels_Benign, Images_Malignant, Labels_Malignant]

Score = ConfigurationModels(MainKeys, ModelValues, GeneralDataModels, GeneralDataModelsEsp)

updateCSV(Score, df, column_names, path, 2)
