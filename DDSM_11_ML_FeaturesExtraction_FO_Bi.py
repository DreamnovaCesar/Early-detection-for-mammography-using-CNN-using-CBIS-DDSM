import pandas as pd

from DDSM_2_Folders import DataModelsML_FO
from DDSM_2_Folders import DataModelsML_SO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from DDSM_2_Folders import DataCSV

from DDSM_10_ML_Functions import SVM
from DDSM_10_ML_Functions import MultiSVM
from DDSM_10_ML_Functions import MLP
from DDSM_10_ML_Functions import KNN
from DDSM_10_ML_Functions import RF
from DDSM_10_ML_Functions import DT
from DDSM_10_ML_Functions import GBC

from DDSM_10_ML_Functions import DataModelsExtract
from DDSM_10_ML_Functions import MLConfigurationModels
from DDSM_10_ML_Functions import updateCSV
from DDSM_10_ML_Functions import AllModelValues

from DDSM_3_General_Functions import fos

from DDSM_8_Preprocessing_9_Parameters import labels_Biclass

from DDSM_8_Preprocessing_9_Parameters import RAWTechnique
from DDSM_8_Preprocessing_9_Parameters import NOTechnique
from DDSM_8_Preprocessing_9_Parameters import CLAHETechnique
from DDSM_8_Preprocessing_9_Parameters import HETechnique
from DDSM_8_Preprocessing_9_Parameters import UMTechnique
from DDSM_8_Preprocessing_9_Parameters import CSTechnique

from DDSM_10_ML_DataAugmentation_Bi import Images_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import Images_Malignant
from DDSM_10_ML_DataAugmentation_Bi import Labels_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import Labels_Malignant

from DDSM_10_ML_DataAugmentation_Bi import NOImages_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import NOImages_Malignant
from DDSM_10_ML_DataAugmentation_Bi import NOLabels_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import NOLabels_Malignant

from DDSM_10_ML_DataAugmentation_Bi import CLAHEImages_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import CLAHEImages_Malignant
from DDSM_10_ML_DataAugmentation_Bi import CLAHELabels_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import CLAHELabels_Malignant

from DDSM_10_ML_DataAugmentation_Bi import HEImages_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import HEImages_Malignant
from DDSM_10_ML_DataAugmentation_Bi import HELabels_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import HELabels_Malignant

from DDSM_10_ML_DataAugmentation_Bi import UMImages_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import UMImages_Malignant
from DDSM_10_ML_DataAugmentation_Bi import UMLabels_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import UMLabels_Malignant

from DDSM_10_ML_DataAugmentation_Bi import CSImages_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import CSImages_Malignant
from DDSM_10_ML_DataAugmentation_Bi import CSLabels_BenignWC
from DDSM_10_ML_DataAugmentation_Bi import CSLabels_Malignant

from DDSM_4_DDSM_Functions import TexturesFeatureFirstOrderImage
from DDSM_4_DDSM_Functions import TexturesFeatureGLCMImage
from DDSM_4_DDSM_Functions import TexturesFeatureGLRLMImage

from DDSM_10_ML_Functions import ModelsML

df = pd.read_csv("D:\DDSM\Calc Mammography\DataCSV\DataFrame_Binary_DDSM_Data_ML_FO.csv")
path = "D:\DDSM\Calc Mammography\DataCSV\DataFrame_Binary_DDSM_Data_ML_FO.csv"

MainKeys = ['Model function', 'Technique', 'Labels','Images 1', 'Labels 1', 'Images 2', 'Labels 2']
MainKeys_Extract = ['Images 1', 'Labels 1', 'Images 2', 'Labels 2']

column_names = ["Model used", "Accuracy", "Precision", "Recall", "F1_Score", "Training images", "Test images", "Time training", "Technique", "TN", "FP", "FN", "TP", "AUC"]

ModelValuesRAW = [labels_Biclass, Images_BenignWC, Labels_BenignWC, Images_Malignant, Labels_Malignant]
ModelValuesNO = [labels_Biclass, NOImages_BenignWC, NOLabels_BenignWC, NOImages_Malignant, NOLabels_Malignant]
ModelValuesCLAHE = [labels_Biclass, CLAHEImages_BenignWC, CLAHELabels_BenignWC, CLAHEImages_Malignant, CLAHELabels_Malignant]
ModelValuesHE = [labels_Biclass, HEImages_BenignWC, HELabels_BenignWC, HEImages_Malignant, HELabels_Malignant]
ModelValuesUM = [labels_Biclass, UMImages_BenignWC, UMLabels_BenignWC, UMImages_Malignant, UMLabels_Malignant]
ModelValuesCS = [labels_Biclass, CSImages_BenignWC, CSLabels_BenignWC, CSImages_Malignant, CSLabels_Malignant]

ModelML = SVM

MultiSVMConfigRAW = [ModelML, RAWTechnique, labels_Biclass]
MultiSVMConfigNO = [ModelML, NOTechnique, labels_Biclass]
MultiSVMConfigCLAHE = [ModelML, CLAHETechnique, labels_Biclass]
MultiSVMConfigHE = [ModelML, HETechnique, labels_Biclass]
MultiSVMConfigUM = [ModelML, UMTechnique, labels_Biclass]
MultiSVMConfigCS = [ModelML, CSTechnique, labels_Biclass]

ModelML = MLP

MLPConfigRAW = [ModelML, RAWTechnique, labels_Biclass]
MLPConfigNO = [ModelML, NOTechnique, labels_Biclass]
MLPConfigCLAHE = [ModelML, CLAHETechnique, labels_Biclass]
MLPConfigHE = [ModelML, HETechnique, labels_Biclass]
MLPConfigUM = [ModelML, UMTechnique, labels_Biclass]
MLPConfigCS = [ModelML, CSTechnique, labels_Biclass]

ModelML = KNN

KNNConfigRAW = [ModelML, RAWTechnique, labels_Biclass]
KNNConfigNO = [ModelML, NOTechnique, labels_Biclass]
KNNConfigCLAHE = [ModelML, CLAHETechnique, labels_Biclass]
KNNConfigHE = [ModelML, HETechnique, labels_Biclass]
KNNConfigUM = [ModelML, UMTechnique, labels_Biclass]
KNNConfigCS = [ModelML, CSTechnique, labels_Biclass]

ModelML = RF

RFConfigRAW = [ModelML, RAWTechnique, labels_Biclass]
RFConfigNO = [ModelML, NOTechnique, labels_Biclass]
RFConfigCLAHE = [ModelML, CLAHETechnique, labels_Biclass]
RFConfigHE = [ModelML, HETechnique, labels_Biclass]
RFConfigUM = [ModelML, UMTechnique, labels_Biclass]
RFConfigCS = [ModelML, CSTechnique, labels_Biclass]

ModelML = DT

DTConfigRAW = [ModelML, RAWTechnique, labels_Biclass]
DTConfigNO = [ModelML, NOTechnique, labels_Biclass]
DTConfigCLAHE = [ModelML, CLAHETechnique, labels_Biclass]
DTConfigHE = [ModelML, HETechnique, labels_Biclass]
DTConfigUM = [ModelML, UMTechnique, labels_Biclass]
DTConfigCS = [ModelML, CSTechnique, labels_Biclass]

ModelML = GBC

GBCConfigRAW = [ModelML, RAWTechnique, labels_Biclass]
GBCConfigNO = [ModelML, NOTechnique, labels_Biclass]
GBCConfigCLAHE = [ModelML, CLAHETechnique, labels_Biclass]
GBCConfigHE = [ModelML, HETechnique, labels_Biclass]
GBCConfigUM = [ModelML, UMTechnique, labels_Biclass]
GBCConfigCS = [ModelML, CSTechnique, labels_Biclass]

TotalConfigModels = [MultiSVMConfigRAW, MultiSVMConfigNO, MultiSVMConfigCLAHE, MultiSVMConfigHE, MultiSVMConfigUM, MultiSVMConfigCS,
                    MLPConfigRAW, MLPConfigNO, MLPConfigCLAHE, MLPConfigHE, MLPConfigUM, MLPConfigCS,
                    KNNConfigRAW, KNNConfigNO, KNNConfigCLAHE, KNNConfigHE, KNNConfigUM, KNNConfigCS,
                    RFConfigRAW, RFConfigNO, RFConfigCLAHE, RFConfigHE, RFConfigUM, RFConfigCS,
                    DTConfigRAW, DTConfigNO, DTConfigCLAHE, DTConfigHE, DTConfigUM, DTConfigCS,
                    GBCConfigRAW, GBCConfigNO, GBCConfigCLAHE, GBCConfigHE, GBCConfigUM, GBCConfigCS]

MultiSVMConfigModels = [MultiSVMConfigRAW, MultiSVMConfigNO, MultiSVMConfigCLAHE, MultiSVMConfigHE, MultiSVMConfigUM, MultiSVMConfigCS]

MLPConfigModels = [MLPConfigRAW, MLPConfigNO, MLPConfigCLAHE, MLPConfigHE, MLPConfigUM, MLPConfigCS]

KNNConfigModels = [KNNConfigRAW, KNNConfigNO, KNNConfigCLAHE, KNNConfigHE, KNNConfigUM, KNNConfigCS]

RFConfigModels = [RFConfigRAW, RFConfigNO, RFConfigCLAHE, RFConfigHE, RFConfigUM, RFConfigCS]

DTConfigModels = [DTConfigRAW, DTConfigNO, DTConfigCLAHE, DTConfigHE, DTConfigUM, DTConfigCS]

GBCConfigModels = [GBCConfigRAW, GBCConfigNO, GBCConfigCLAHE, GBCConfigHE, GBCConfigUM, GBCConfigCS]

#TotalModelValues = [ModelValuesRAW, ModelValuesNO, ModelValuesCLAHE, ModelValuesHE, ModelValuesUM, ModelValuesCS]

DataFrameRAW, NameRAW = DataModelsExtract(MainKeys_Extract, ModelValuesRAW, TexturesFeatureFirstOrderImage)

DataFrameNO, NameNO = DataModelsExtract(MainKeys_Extract, ModelValuesNO, TexturesFeatureFirstOrderImage)

DataFrameCLAHE, NameCLAHE = DataModelsExtract(MainKeys_Extract, ModelValuesCLAHE, TexturesFeatureFirstOrderImage)

DataFrameHE, NameHE = DataModelsExtract(MainKeys_Extract, ModelValuesHE, TexturesFeatureFirstOrderImage)

DataFrameUM, NameUM = DataModelsExtract(MainKeys_Extract, ModelValuesUM, TexturesFeatureFirstOrderImage)

DataFrameCS, NameCS = DataModelsExtract(MainKeys_Extract, ModelValuesCS, TexturesFeatureFirstOrderImage)


AllModelValues(df, column_names, path, DataFrameRAW, MultiSVMConfigModels, DataModelsML_FO, DataCSV, NameRAW, 0)

AllModelValues(df, column_names, path, DataFrameNO, MLPConfigModels, DataModelsML_FO, DataCSV, NameNO, 6)

AllModelValues(df, column_names, path, DataFrameCLAHE, KNNConfigModels, DataModelsML_FO, DataCSV, NameCLAHE, 12)

AllModelValues(df, column_names, path, DataFrameHE, RFConfigModels, DataModelsML_FO, DataCSV, NameHE, 18)

AllModelValues(df, column_names, path, DataFrameUM, DTConfigModels, DataModelsML_FO, DataCSV, NameUM, 24)

AllModelValues(df, column_names, path, DataFrameCS, GBCConfigModels, DataModelsML_FO, DataCSV, NameCS, 30)
