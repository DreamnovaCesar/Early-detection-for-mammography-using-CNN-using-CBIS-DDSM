import os
import pandas as pd

from DDSM_5_Processing_Functions import CLAHE_Technique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import DataCSV

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

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

IB = 0  # Normal Images
IM = 1  # Tumor Images

Benign = 'Benign'
Malignant = 'Malignant'

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Images_CLAHE1_DDSM, DataFrame_CLAHE_Data1_DDSM = CLAHE_Technique(NOALLBenignImages, CLAHEALLBenignImages, Benign, Clip_limit, IB)
Images_CLAHE3_DDSM, DataFrame_CLAHE_Data3_DDSM = CLAHE_Technique(NOMalignantImages, CLAHEMalignantImages, Malignant, Clip_limit, IM)

DataFrameCLAHE_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameCLAHE_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_CLAHE_DDSM_Bi' + '.csv'
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

Images_CLAHE1_DDSM, DataFrame_CLAHE_Data1_DDSM = CLAHE_Technique(NOBenignImages, CLAHEBenignImages, Benign, Clip_limit, IB)
Images_CLAHE2_DDSM, DataFrame_CLAHE_Data2_DDSM = CLAHE_Technique(NOBenignWCImages, CLAHEBenignWCImages, BenignWC, Clip_limit, IBWC)
Images_CLAHE3_DDSM, DataFrame_CLAHE_Data3_DDSM = CLAHE_Technique(NOMalignantImages, CLAHEMalignantImages, Malignant, Clip_limit, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data2_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_CLAHE_DDSM_Multi_All' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

IB = 0  # Normal Images
IM = 1  # Normal Images

Images_CLAHE1_DDSM, DataFrame_CLAHE_Data1_DDSM = CLAHE_Technique(NOBenignImages, CLAHEBenignImages, Benign, Clip_limit, IB)
Images_CLAHE2_DDSM, DataFrame_CLAHE_Data2_DDSM = CLAHE_Technique(NOAbnormalImages, CLAHEAbnormalImages, Malignant, Clip_limit, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data2_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_CLAHE_DDSM_Multi_Abnormal' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########