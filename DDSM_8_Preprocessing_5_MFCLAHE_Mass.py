import os
import pandas as pd

from DDSM_5_Processing_Functions import CLAHE_Technique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import DataCSVMass

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import NOALLBenignMassImages
from DDSM_2_Folders import NOBenignMassImages
from DDSM_2_Folders import NOBenignWCMassImages
from DDSM_2_Folders import NOMalignantMassImages
from DDSM_2_Folders import NOAbnormalMassImages

from DDSM_2_Folders import CLAHEALLBenignMassImages
from DDSM_2_Folders import CLAHEBenignMassImages
from DDSM_2_Folders import CLAHEBenignWCMassImages
from DDSM_2_Folders import CLAHEMalignantMassImages
from DDSM_2_Folders import CLAHEAbnormalMassImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Clip_limit = 0.01

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

IB = 0  # Normal Images
IM = 1  # Tumor Images

Benign = 'Benign'
Malignant = 'Malignant'

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Images_CLAHE1_DDSM, DataFrame_CLAHE_Data1_DDSM = CLAHE_Technique(NOALLBenignMassImages, CLAHEALLBenignMassImages, Benign, Clip_limit, IB)
Images_CLAHE3_DDSM, DataFrame_CLAHE_Data3_DDSM = CLAHE_Technique(NOMalignantMassImages, CLAHEMalignantMassImages, Malignant, Clip_limit, IM)

DataFrameCLAHE_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameCLAHE_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_CLAHE_DDSM_Mass_bi' + '.csv'
dstPath = os.path.join(DataCSVMass, dst)

print(DataFrameCLAHE_ALL_DDSM)
DataFrameCLAHE_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

IB = 0  # Normal Images
IBWC = 1  # Normal Images
IM = 2  # Tumor Images

Benign = 'Benign'
BenignWC = 'BenignWC'
Malignant = 'Malignant'

Images_CLAHE1_DDSM, DataFrame_CLAHE_Data1_DDSM = CLAHE_Technique(NOBenignMassImages, CLAHEBenignMassImages, Benign, Clip_limit, IB)
Images_CLAHE2_DDSM, DataFrame_CLAHE_Data2_DDSM = CLAHE_Technique(NOBenignWCMassImages, CLAHEBenignWCMassImages, BenignWC, Clip_limit, IBWC)
Images_CLAHE4_DDSM, DataFrame_CLAHE_Data4_DDSM = CLAHE_Technique(NOMalignantMassImages, CLAHEMalignantMassImages, Malignant, Clip_limit, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data2_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_CLAHE_DDSM_Multi_Mass_All' + '.csv'
dstPath = os.path.join(DataCSVMass, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

IB = 0  # Normal Images
IM = 1  # Normal Images

Images_CLAHE1_DDSM, DataFrame_CLAHE_Data1_DDSM = CLAHE_Technique(NOBenignMassImages, CLAHEBenignMassImages, Benign, Clip_limit, IB)
Images_CLAHE2_DDSM, DataFrame_CLAHE_Data2_DDSM = CLAHE_Technique(NOAbnormalMassImages, CLAHEAbnormalMassImages, Malignant, Clip_limit, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data2_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_CLAHE_DDSM_Multi_Mass_Abnormal' + '.csv'
dstPath = os.path.join(DataCSVMass, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########