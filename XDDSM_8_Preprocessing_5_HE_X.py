import os
import pandas as pd

from DDSM_5_Processing_Functions import HE_Technique

from DDSM_2_Folders import DataCSV

from DDSM_2_Folders import NOALLBenignImages
from DDSM_2_Folders import NOBenignImages
from DDSM_2_Folders import NOBenignWCImages
from DDSM_2_Folders import NOMalignantImages

from DDSM_2_Folders import MFALLBenignImages
from DDSM_2_Folders import MFBenignImages
from DDSM_2_Folders import MFBenignWCImages
from DDSM_2_Folders import MFMalignantImages

from DDSM_2_Folders import HEALLBenignImages
from DDSM_2_Folders import HEBenignImages
from DDSM_2_Folders import HEBenignWCImages
from DDSM_2_Folders import HEMalignantImages


IB = 0  # Normal Images
IM = 1  # Tumor Images

Benign = 'Benign'
Malignant = 'Malignant'

Images_HE1_DDSM, DataFrame_HE_Data1_DDSM = HE_Technique(NOALLBenignImages, HEALLBenignImages, Benign, IB)
Images_HE2_DDSM, DataFrame_HE_Data2_DDSM = HE_Technique(NOMalignantImages, HEMalignantImages, Malignant, IM)

DataFrameHE_ALL_DDSM = pd.concat([DataFrame_HE_Data1_DDSM, DataFrame_HE_Data2_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameHE_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_HE_DDSM_Info' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameHE_ALL_DDSM)
DataFrameHE_ALL_DDSM.to_csv(dstPath)

IB = 0  # Normal Images
IBWC = 1  # Normal Images
IM = 2  # Tumor Images

Benign = 'Benign'
BenignWC = 'BenignWC'
Malignant = 'Malignant'

Images_HE1_DDSM, DataFrame_HE_Data1_DDSM = HE_Technique(NOBenignImages, HEBenignImages, Benign, IB)
Images_HE2_DDSM, DataFrame_HE_Data2_DDSM = HE_Technique(NOBenignWCImages, HEBenignWCImages, BenignWC, IBWC)
Images_HE3_DDSM, DataFrame_HE_Data3_DDSM = HE_Technique(NOMalignantImages, HEMalignantImages, Malignant, IM)

DataFrameHE_ALL_DDSM = pd.concat([DataFrame_HE_Data1_DDSM, DataFrame_HE_Data2_DDSM, DataFrame_HE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameHE_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_HE_DDSM_Multi_Info' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameHE_ALL_DDSM)
DataFrameHE_ALL_DDSM.to_csv(dstPath)