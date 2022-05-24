
import pandas as pd

from DDSM_2_Folders import GeneralDataModels
from DDSM_2_Folders import GeneralDataModelsEsp

from DDSM_8_Preprocessing_9_Parameters import labels_Biclass

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

df = pd.read_csv("D:\DDSM\DataCSV\Biclass_DataFrame_DDSM_Data.csv")
path = "D:\DDSM\DataCSV\Biclass_DataFrame_MIAS_Data.csv"

MainKeys = ['Model function', 'Technique', 'Labels', 'Xsize', 'Ysize', 'Validation split', 'Epochs','Images 1', 'Labels 1', 'Images 2', 'Labels 2']
column_names = ['Name Model', "Model used", "Accuracy Training FE", "Accuracy Training LE", "Accuracy Testing", "Loss Train", "Loss Test", "Training images", "Test images", "Precision", "Recall", "F1_Score", "Time training", "Time testing", "Technique used", "TN", "FP", "FN", "TP", "Epochs","Auc"]

ModelTest = MobileNetV3Large_Pretrained

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import HEImages_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import HELabels_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import HEImages_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import HELabels_Mass

ModelValues =  [ModelTest, HETechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, HEImages_Calcification, HELabels_Calcification, HEImages_Mass, HELabels_Mass]

Score = ConfigurationModels(MainKeys, ModelValues, GeneralDataModels, GeneralDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 9)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import UMImages_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import UMLabels_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import UMImages_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import UMLabels_Mass

ModelValues1 =  [ModelTest, UMTechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, UMImages_Calcification, UMLabels_Calcification, UMImages_Mass, UMLabels_Mass]

Score = ConfigurationModels(MainKeys, ModelValues1, GeneralDataModels, GeneralDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 10)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import CSImages_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import CSLabels_Calcification
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import CSImages_Mass
from DDSM_8_Preprocessing_8_DataAugmentation_Mass_Calc_Bi_2 import CSLabels_Mass

ModelValues2 =  [ModelTest, CSTechnique, labels_Biclass, XsizeResized, YsizeResized, Valid_split, Epochs, CSImages_Calcification, CSLabels_Calcification, CSImages_Mass, CSLabels_Mass]

Score = ConfigurationModels(MainKeys, ModelValues2, GeneralDataModels, GeneralDataModelsEsp)

UpdateCSV(Score, df, column_names, path, 11)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

