import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from DDSM_2_Folders import GeneralMultiDataModels
from DDSM_2_Folders import GeneralMultiDataModelsEsp

from DDSM_8_Preprocessing_9_Parameters import labels_Triclass

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

from DDSM_4_DDSM_Functions import ConfigurationModels
from DDSM_4_DDSM_Functions import UpdateCSV

df = pd.read_csv("D:\DDSM\MultiDataCSV\Multiclass_DataFrame_DDSM_Data.csv")
path = "D:\DDSM\MultiDataCSV\Multiclass_DataFrame_DDSM_Data.csv"

MainKeys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs','Images 1', 'Labels 1', 'Images 2', 'Labels 2']
column_names = ['Name Model', "Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs", "Auc Normal", "Auc Benign", "Auc Malignant"]

ModelTest = CustomCNNAlexNet12_Model
"""
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import HEImages_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import HELabels_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import HEImages_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import HELabels_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import HEImages_Normal
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import HELabels_Normal

ModelValues =   [ModelTest, HETechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, HEImages_Normal, HELabels_Normal, HEImages_Mass, HELabels_Mass, HEImages_Calcification, HELabels_Calcification]

Score = ConfigurationModels(MainKeys, ModelValues, GeneralMultiDataModels, GeneralMultiDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 58)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import UMImages_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import UMLabels_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import UMImages_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import UMLabels_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import UMImages_Normal
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import UMLabels_Normal

ModelValues1 =  [ModelTest, UMTechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, UMImages_Normal, UMLabels_Normal, UMImages_Mass, UMLabels_Mass, UMImages_Calcification, UMLabels_Calcification]

Score = ConfigurationModels(MainKeys, ModelValues1, GeneralMultiDataModels, GeneralMultiDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 59)
"""
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import CSImages_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import CSLabels_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import CSImages_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import CSLabels_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import CSImages_Normal
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Multi_2 import CSLabels_Normal

ModelValues2 =  [ModelTest, CSTechnique, labels_Triclass, XsizeResized, YsizeResized, Valid_split, Epochs, CSImages_Normal, CSLabels_Normal, CSImages_Mass, CSLabels_Mass, CSImages_Calcification, CSLabels_Calcification]

Score = ConfigurationModels(MainKeys, ModelValues2, GeneralMultiDataModels, GeneralMultiDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 61)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########
