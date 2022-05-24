import os
import pandas as pd

from DDSM_5_Processing_Functions import CLAHE_Technique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import DataCSV
from DDSM_2_Folders import MultiDataCSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import NOALLBenignImages
from DDSM_2_Folders import NOBenignImages
from DDSM_2_Folders import NOBenignWCImages
from DDSM_2_Folders import NOMalignantImages
from DDSM_2_Folders import NOAbnormalImages

from DDSM_2_Folders import CLAHEALLBenignImages
from DDSM_2_Folders import CLAHEBenignImages
from DDSM_2_Folders import CLAHEBenignWCImages
from DDSM_2_Folders import CLAHEMalignantImages
from DDSM_2_Folders import CLAHEAbnormalImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Clip_limit = 0.01

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Tumor

IB = 0  # Normal Images
IM = 1  # Tumor Images

Benign = 'Benign'
Malignant = 'Malignant'

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Tumor

DataFrame_CLAHE_Data1_DDSM = CLAHE_Technique(NOALLBenignImages, CLAHEALLBenignImages, Benign, Clip_limit, IB)
DataFrame_CLAHE_Data3_DDSM = CLAHE_Technique(NOMalignantImages, CLAHEMalignantImages, Malignant, Clip_limit, IM)

DataFrameCLAHE_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameCLAHE_ALL_DDSM.shape[0] + 1)

dst = 'Biclass_Dataframe_CLAHE_Calc' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameCLAHE_ALL_DDSM)
DataFrameCLAHE_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Tumor

IB = 0  # Normal Images
IBWC = 1  # Normal Images
IM = 2  # Tumor Images

Benign = 'Benign'
BenignWC = 'BenignWC'
Malignant = 'Malignant'
Abnormal = 'Abnormal'

DataFrame_CLAHE_Data1_DDSM = CLAHE_Technique(NOBenignImages, CLAHEBenignImages, Benign, Clip_limit, IB)
DataFrame_CLAHE_Data2_DDSM = CLAHE_Technique(NOBenignWCImages, CLAHEBenignWCImages, BenignWC, Clip_limit, IBWC)
DataFrame_CLAHE_Data3_DDSM = CLAHE_Technique(NOMalignantImages, CLAHEMalignantImages, Malignant, Clip_limit, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data2_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'Multiclass_Dataframe_CLAHE_Calc' + '.csv'
dstPath = os.path.join(MultiDataCSV, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########
