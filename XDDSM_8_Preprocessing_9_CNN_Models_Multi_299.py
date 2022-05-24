import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from DDSM_2_Folders import MultiDataModels
from DDSM_2_Folders import MultiDataModelsEsp

from DDSM_8_Preprocessing_9_Parameters import labels_Triclass

from DDSM_8_Preprocessing_9_Parameters import RAWTechnique
from DDSM_8_Preprocessing_9_Parameters import NOTechnique
from DDSM_8_Preprocessing_9_Parameters import CLAHETechnique
from DDSM_8_Preprocessing_9_Parameters import HETechnique
from DDSM_8_Preprocessing_9_Parameters import UMTechnique
from DDSM_8_Preprocessing_9_Parameters import CSTechnique

from DDSM_8_Preprocessing_9_Parameters import Valid_split
from DDSM_8_Preprocessing_9_Parameters import Epochs

from DDSM_8_Preprocessing_9_Parameters import XsizeResized299
from DDSM_8_Preprocessing_9_Parameters import YsizeResized299
"""
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import Images_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import Labels_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import Images_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import Labels_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import Images_Malignant
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import Labels_Malignant
"""
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import NOImages_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import NOLabels_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import NOImages_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import NOLabels_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import NOImages_Malignant
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import NOLabels_Malignant
"""
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import CLAHEImages_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import CLAHELabels_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import CLAHEImages_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import CLAHELabels_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import CLAHEImages_Malignant

from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import HEImages_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import HELabels_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import HEImages_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import HELabels_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import HEImages_Malignant
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import HELabels_Malignant

from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import UMImages_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import UMLabels_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import UMImages_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import UMLabels_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import UMImages_Malignant
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import UMLabels_Malignant

from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import CSImages_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import CSLabels_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import CSImages_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import CSLabels_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import CSImages_Malignant
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import CSLabels_Malignant

from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import MFCLAHEImages_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import MFCLAHELabels_Benign
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import MFCLAHEImages_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import MFCLAHELabels_BenignWC
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import MFCLAHEImages_Malignant
from DDSM_8_Preprocessing_8_DataAugmentation_Multi_299 import MFCLAHELabels_Malignant
"""
from DDSM_7_1_CNN_Architectures import InceptionResNetV2_PreTrained, PreTrainedModels
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

def updateCSV(df, column_names, path, row):

    print(df.head(len(df.index)))

    for i in range(len(Score)):
        df.loc[row, column_names[i]] = Score[i]
    
    print(df.head(len(df.index)))

    df.to_csv(path, index = False)
  
    print(df)

df = pd.read_csv("D:\DDSM\Calc Mammography\Calcification299\MultiDataCSV299\DataFrame_Multi_DDSM_Data_299.csv")
path = "D:\DDSM\Calc Mammography\Calcification299\MultiDataCSV299\DataFrame_Multi_DDSM_Data_299.csv"

MainKeys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs','Images 1', 'Labels 1', 'Images 2', 'Labels 2']
column_names = ["Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs", "Auc Normal", "Auc Benign", "Auc Malignant"]

ModelTest = InceptionResNetV2_PreTrained

#ModelValues =   [ModelTest, RAWTechnique, labels_Triclass, XsizeResized299, YsizeResized299, Valid_split, Epochs, Images_Benign, Labels_Benign, Images_BenignWC, Labels_BenignWC, Images_Malignant, Labels_Malignant]
ModelValues1 =  [ModelTest, NOTechnique, labels_Triclass, XsizeResized299, YsizeResized299, Valid_split, Epochs, NOImages_Benign, NOLabels_Benign, NOImages_BenignWC, NOLabels_BenignWC, NOImages_Malignant, NOLabels_Malignant]

#ModelValues2 =  [ModelTest, CLAHETechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, CLAHEImages_Benign, CLAHELabels_Benign, CLAHEImages_BenignWC, CLAHELabels_BenignWC, CLAHEImages_Malignant, CLAHELabels_Malignant]
#ModelValues3 =  [ModelTest, HETechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, HEImages_Benign, HELabels_Benign, HEImages_BenignWC, HELabels_BenignWC, HEImages_Malignant, HELabels_Malignant]

#ModelValues4 =  [ModelTest, UMTechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, UMImages_Benign, UMLabels_Benign, UMImages_BenignWC, UMLabels_BenignWC, UMImages_Malignant, UMLabels_Malignant]
#ModelValues5 =  [ModelTest, CSTechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, CSImages_Benign, CSLabels_Benign, CSImages_BenignWC, CSLabels_BenignWC, CSImages_Malignant, CSLabels_Malignant]

Score = ConfigurationModels(MainKeys, ModelValues1, MultiDataModels, MultiDataModelsEsp)

updateCSV(df, column_names, path, 0)

