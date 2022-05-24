import os
import pandas as pd

from DDSM_5_Processing_Functions import HE_Technique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import DataCSVMass
from DDSM_2_Folders import MultiDataCSVMass

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import NOALLBenignMassImages
from DDSM_2_Folders import NOBenignMassImages
from DDSM_2_Folders import NOBenignWCMassImages
from DDSM_2_Folders import NOMalignantMassImages
from DDSM_2_Folders import NOAbnormalMassImages

from DDSM_2_Folders import HEALLBenignMassImages
from DDSM_2_Folders import HEBenignMassImages
from DDSM_2_Folders import HEBenignWCMassImages
from DDSM_2_Folders import HEMalignantMassImages
from DDSM_2_Folders import HEAbnormalMassImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

IB = 0  # Normal Images
IM = 1  # Tumor Images

Benign = 'Benign'
Malignant = 'Malignant'

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

DataFrame_CLAHE_Data1_DDSM = HE_Technique(NOALLBenignMassImages, HEALLBenignMassImages, Benign, IB)
DataFrame_CLAHE_Data3_DDSM = HE_Technique(NOMalignantMassImages, HEMalignantMassImages, Malignant, IM)

DataFrameCLAHE_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameCLAHE_ALL_DDSM.shape[0] + 1)

dst = 'Biclass_Dataframe_HE_Mass' + '.csv'
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
Abnormal = 'Abnormal'

DataFrame_CLAHE_Data1_DDSM = HE_Technique(NOBenignMassImages, HEBenignMassImages, Benign, IB)
DataFrame_CLAHE_Data2_DDSM = HE_Technique(NOBenignWCMassImages, HEBenignWCMassImages, BenignWC, IBWC)
DataFrame_CLAHE_Data4_DDSM = HE_Technique(NOMalignantMassImages, HEMalignantMassImages, Malignant, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data2_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'Multiclass_Dataframe_HE_Mass' + '.csv'
dstPath = os.path.join(MultiDataCSVMass, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########