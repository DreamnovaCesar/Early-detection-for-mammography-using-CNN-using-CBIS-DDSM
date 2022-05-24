import os
import pandas as pd

from DDSM_5_Processing_Functions import HE_Technique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import DataCSV
from DDSM_2_Folders import MultiDataCSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import NOALLBenignImages
from DDSM_2_Folders import NOBenignImages
from DDSM_2_Folders import NOBenignWCImages
from DDSM_2_Folders import NOMalignantImages
from DDSM_2_Folders import NOAbnormalImages

from DDSM_2_Folders import HEALLBenignImages
from DDSM_2_Folders import HEBenignImages
from DDSM_2_Folders import HEBenignWCImages
from DDSM_2_Folders import HEMalignantImages
from DDSM_2_Folders import HEAbnormalImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Clip_limit = 0.01

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

IB = 0  # Normal Images
IM = 1  # Tumor Images

Benign = 'Benign'
Malignant = 'Malignant'

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

DataFrame_CLAHE_Data1_DDSM = HE_Technique(NOALLBenignImages, HEALLBenignImages, Benign, IB)
DataFrame_CLAHE_Data3_DDSM = HE_Technique(NOMalignantImages, HEMalignantImages, Malignant, IM)

DataFrameCLAHE_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameCLAHE_ALL_DDSM.shape[0] + 1)

dst = 'Biclass_Dataframe_HE_Calc' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameCLAHE_ALL_DDSM)
DataFrameCLAHE_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

IB = 0  # Normal Images
IBWC = 1  # Normal Images
IM = 2  # Tumor Images

Benign = 'Benign'
BenignWC = 'BenignWC'
Malignant = 'Malignant'
Abnormal = 'Abnormal'

DataFrame_CLAHE_Data1_DDSM = HE_Technique(NOBenignImages, HEBenignImages, Benign, IB)
DataFrame_CLAHE_Data2_DDSM = HE_Technique(NOBenignWCImages, HEBenignWCImages, BenignWC, IBWC)
DataFrame_CLAHE_Data3_DDSM = HE_Technique(NOMalignantImages, HEMalignantImages, Malignant, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data2_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'Multiclass_Dataframe_HE_Calc' + '.csv'
dstPath = os.path.join(MultiDataCSV, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########
