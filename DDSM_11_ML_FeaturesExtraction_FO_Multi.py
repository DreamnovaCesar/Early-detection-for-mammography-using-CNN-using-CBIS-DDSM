import pandas as pd

from DDSM_2_Folders import MultiDataModelsML_FO
from DDSM_2_Folders import MultiDataModelsML_SO

from DDSM_2_Folders import MultiDataCSV

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

from DDSM_8_Preprocessing_9_Parameters import labels_Triclass

from DDSM_8_Preprocessing_9_Parameters import RAWTechnique
from DDSM_8_Preprocessing_9_Parameters import NOTechnique
from DDSM_8_Preprocessing_9_Parameters import CLAHETechnique
from DDSM_8_Preprocessing_9_Parameters import HETechnique
from DDSM_8_Preprocessing_9_Parameters import UMTechnique
from DDSM_8_Preprocessing_9_Parameters import CSTechnique

from DDSM_10_ML_DataAugmentation_Multi import Images_Benign
from DDSM_10_ML_DataAugmentation_Multi import Labels_Benign
from DDSM_10_ML_DataAugmentation_Multi import Images_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import Labels_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import Images_Malignant
from DDSM_10_ML_DataAugmentation_Multi import Labels_Malignant

from DDSM_10_ML_DataAugmentation_Multi import NOImages_Benign
from DDSM_10_ML_DataAugmentation_Multi import NOLabels_Benign
from DDSM_10_ML_DataAugmentation_Multi import NOImages_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import NOLabels_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import NOImages_Malignant
from DDSM_10_ML_DataAugmentation_Multi import NOLabels_Malignant

from DDSM_10_ML_DataAugmentation_Multi import CLAHEImages_Benign
from DDSM_10_ML_DataAugmentation_Multi import CLAHELabels_Benign
from DDSM_10_ML_DataAugmentation_Multi import CLAHEImages_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import CLAHELabels_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import CLAHEImages_Malignant
from DDSM_10_ML_DataAugmentation_Multi import CLAHELabels_Malignant

from DDSM_10_ML_DataAugmentation_Multi import HEImages_Benign
from DDSM_10_ML_DataAugmentation_Multi import HELabels_Benign
from DDSM_10_ML_DataAugmentation_Multi import HEImages_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import HELabels_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import HEImages_Malignant
from DDSM_10_ML_DataAugmentation_Multi import HELabels_Malignant

from DDSM_10_ML_DataAugmentation_Multi import UMImages_Benign
from DDSM_10_ML_DataAugmentation_Multi import UMLabels_Benign
from DDSM_10_ML_DataAugmentation_Multi import UMImages_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import UMLabels_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import UMImages_Malignant
from DDSM_10_ML_DataAugmentation_Multi import UMLabels_Malignant

from DDSM_10_ML_DataAugmentation_Multi import CSImages_Benign
from DDSM_10_ML_DataAugmentation_Multi import CSLabels_Benign
from DDSM_10_ML_DataAugmentation_Multi import CSImages_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import CSLabels_BenignWC
from DDSM_10_ML_DataAugmentation_Multi import CSImages_Malignant
from DDSM_10_ML_DataAugmentation_Multi import CSLabels_Malignant

from DDSM_4_DDSM_Functions import TexturesFeatureFirstOrderImage
from DDSM_4_DDSM_Functions import TexturesFeatureGLCMImage
from DDSM_4_DDSM_Functions import TexturesFeatureGLRLMImage

from DDSM_10_ML_Functions import ModelsML

from imblearn.over_sampling import SMOTE
sm = SMOTE()

df = pd.read_csv("D:\DDSM\Calc Mammography\MultiDataCSV\DataFrame_Multi_DDSM_Data_ML_FO.csv")
path = "D:\DDSM\Calc Mammography\MultiDataCSV\DataFrame_Multi_DDSM_Data_ML_FO.csv"

MainKeys = ['Model function', 'Technique', 'Labels','Images 1', 'Labels 1', 'Images 2', 'Labels 2']
MainKeys_Extract = ['Images 1', 'Labels 1', 'Images 2', 'Labels 2']

column_names = ["Model used", "Accuracy", "Precision", "Recall", "F1_Score", "Training images", "Test images", "Time training", "Technique", "TN", "FP", "FN", "TP", "Auc Normal", "Auc Benign", "Auc Malignant"]

ModelValuesRAW = [labels_Triclass, Images_Benign, Labels_Benign, Images_BenignWC, Labels_BenignWC, Images_Malignant, Labels_Malignant]
ModelValuesNO = [labels_Triclass, NOImages_Benign, NOLabels_Benign, NOImages_BenignWC, NOLabels_BenignWC, NOImages_Malignant, NOLabels_Malignant]
ModelValuesCLAHE = [labels_Triclass, CLAHEImages_Benign, CLAHELabels_Benign, CLAHEImages_BenignWC, CLAHELabels_BenignWC, CLAHEImages_Malignant, CLAHELabels_Malignant]
ModelValuesHE = [labels_Triclass, HEImages_Benign, HELabels_Benign, HEImages_BenignWC, HELabels_BenignWC, HEImages_Malignant, HELabels_Malignant]
ModelValuesUM = [labels_Triclass, UMImages_Benign, UMLabels_Benign, UMImages_BenignWC, UMLabels_BenignWC, UMImages_Malignant, UMLabels_Malignant]
ModelValuesCS = [labels_Triclass, CSImages_Benign, CSLabels_Benign, CSImages_BenignWC, CSLabels_BenignWC, CSImages_Malignant, CSLabels_Malignant]

ModelML = MultiSVM

MultiSVMConfigRAW = [ModelML, RAWTechnique, labels_Triclass]
MultiSVMConfigNO = [ModelML, NOTechnique, labels_Triclass]
MultiSVMConfigCLAHE = [ModelML, CLAHETechnique, labels_Triclass]
MultiSVMConfigHE = [ModelML, HETechnique, labels_Triclass]
MultiSVMConfigUM = [ModelML, UMTechnique, labels_Triclass]
MultiSVMConfigCS = [ModelML, CSTechnique, labels_Triclass]

ModelML = MLP

MLPConfigRAW = [ModelML, RAWTechnique, labels_Triclass]
MLPConfigNO = [ModelML, NOTechnique, labels_Triclass]
MLPConfigCLAHE = [ModelML, CLAHETechnique, labels_Triclass]
MLPConfigHE = [ModelML, HETechnique, labels_Triclass]
MLPConfigUM = [ModelML, UMTechnique, labels_Triclass]
MLPConfigCS = [ModelML, CSTechnique, labels_Triclass]

ModelML = KNN

KNNConfigRAW = [ModelML, RAWTechnique, labels_Triclass]
KNNConfigNO = [ModelML, NOTechnique, labels_Triclass]
KNNConfigCLAHE = [ModelML, CLAHETechnique, labels_Triclass]
KNNConfigHE = [ModelML, HETechnique, labels_Triclass]
KNNConfigUM = [ModelML, UMTechnique, labels_Triclass]
KNNConfigCS = [ModelML, CSTechnique, labels_Triclass]

ModelML = RF

RFConfigRAW = [ModelML, RAWTechnique, labels_Triclass]
RFConfigNO = [ModelML, NOTechnique, labels_Triclass]
RFConfigCLAHE = [ModelML, CLAHETechnique, labels_Triclass]
RFConfigHE = [ModelML, HETechnique, labels_Triclass]
RFConfigUM = [ModelML, UMTechnique, labels_Triclass]
RFConfigCS = [ModelML, CSTechnique, labels_Triclass]

ModelML = DT

DTConfigRAW = [ModelML, RAWTechnique, labels_Triclass]
DTConfigNO = [ModelML, NOTechnique, labels_Triclass]
DTConfigCLAHE = [ModelML, CLAHETechnique, labels_Triclass]
DTConfigHE = [ModelML, HETechnique, labels_Triclass]
DTConfigUM = [ModelML, UMTechnique, labels_Triclass]
DTConfigCS = [ModelML, CSTechnique, labels_Triclass]

ModelML = GBC

GBCConfigRAW = [ModelML, RAWTechnique, labels_Triclass]
GBCConfigNO = [ModelML, NOTechnique, labels_Triclass]
GBCConfigCLAHE = [ModelML, CLAHETechnique, labels_Triclass]
GBCConfigHE = [ModelML, HETechnique, labels_Triclass]
GBCConfigUM = [ModelML, UMTechnique, labels_Triclass]
GBCConfigCS = [ModelML, CSTechnique, labels_Triclass]

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


AllModelValues(df, column_names, path, DataFrameRAW, MultiSVMConfigModels, MultiDataModelsML_FO, MultiDataCSV, NameRAW, 0)

AllModelValues(df, column_names, path, DataFrameNO, MLPConfigModels, MultiDataModelsML_FO, MultiDataCSV, NameNO, 6)

AllModelValues(df, column_names, path, DataFrameCLAHE, KNNConfigModels, MultiDataModelsML_FO, MultiDataCSV, NameCLAHE, 12)

AllModelValues(df, column_names, path, DataFrameHE, RFConfigModels, MultiDataModelsML_FO, MultiDataCSV, NameHE, 18)

AllModelValues(df, column_names, path, DataFrameUM, DTConfigModels, MultiDataModelsML_FO, MultiDataCSV, NameUM, 24)

AllModelValues(df, column_names, path, DataFrameCS, GBCConfigModels, MultiDataModelsML_FO, MultiDataCSV, NameCS, 30)
