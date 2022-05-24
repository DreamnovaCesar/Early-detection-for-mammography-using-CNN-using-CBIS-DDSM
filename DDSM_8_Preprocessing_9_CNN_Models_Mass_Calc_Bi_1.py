
import pandas as pd

from DDSM_2_Folders import GeneralDataModels
from DDSM_2_Folders import GeneralDataModelsEsp

from DDSM_8_Preprocessing_9_Parameters import labels_Biclass

from DDSM_8_Preprocessing_9_Parameters import RAWTechnique
from DDSM_8_Preprocessing_9_Parameters import NOTechnique
from DDSM_8_Preprocessing_9_Parameters import CLAHETechnique

from DDSM_8_Preprocessing_9_Parameters import Valid_split
from DDSM_8_Preprocessing_9_Parameters import Epochs

from DDSM_8_Preprocessing_9_Parameters import XsizeResized
from DDSM_8_Preprocessing_9_Parameters import YsizeResized

from DDSM_7_1_CNN_Architectures import PreTrainedModels

from DDSM_7_1_CNN_Architectures import MobileNet_Pretrained
from DDSM_7_1_CNN_Architectures import MobileNetV3Small_Pretrained
from DDSM_7_1_CNN_Architectures import MobileNetV3Large_Pretrained

from DDSM_7_1_CNN_Architectures import ResNet50_PreTrained
from DDSM_7_1_CNN_Architectures import ResNet152_PreTrained

from DDSM_7_1_CNN_Architectures import VGG16_PreTrained
from DDSM_7_1_CNN_Architectures import VGG19_PreTrained

from DDSM_7_1_CNN_Architectures import InceptionV3_PreTrained

from DDSM_7_1_CNN_Architectures import DenseNet121_PreTrained
from DDSM_7_1_CNN_Architectures import DenseNet201_PreTrained

from DDSM_7_1_CNN_Architectures import CustomCNNAlexNet12_Model

from DDSM_4_DDSM_Functions import ConfigurationModels
from DDSM_4_DDSM_Functions import UpdateCSV

df = pd.read_csv("D:\DDSM\DataCSV\Biclass_DataFrame_DDSM_Data.csv")
path = "D:\DDSM\DataCSV\Biclass_DataFrame_MIAS_Data.csv"

MainKeys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs','Images 1', 'Labels 1', 'Images 2', 'Labels 2']
column_names = ['Name Model', "Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs","Auc"]

ModelTest = MobileNetV3Small_Pretrained

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import Images_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import Labels_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import Images_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import Labels_Mass

ModelValues =   [ModelTest, RAWTechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, Images_Calcification, Labels_Calcification, Images_Mass, Labels_Mass]

Score = ConfigurationModels(MainKeys, ModelValues, GeneralDataModels, GeneralDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 0)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import NOImages_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import NOLabels_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import NOImages_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import NOLabels_Mass

ModelValues1 =  [ModelTest, NOTechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, NOImages_Calcification, NOLabels_Calcification, NOImages_Mass, NOLabels_Mass]

Score = ConfigurationModels(MainKeys, ModelValues1, GeneralDataModels, GeneralDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 1)

"""
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import CLAHEImages_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import CLAHELabels_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import CLAHEImages_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_1 import CLAHELabels_Mass

ModelValues2 =  [ModelTest, CLAHETechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, CLAHEImages_Calcification, CLAHELabels_Calcification, CLAHEImages_Mass, CLAHELabels_Mass]

Score = ConfigurationModels(MainKeys, ModelValues2, GeneralDataModels, GeneralDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 2)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########
"""
