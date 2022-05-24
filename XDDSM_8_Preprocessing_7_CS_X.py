import os
import pandas as pd

from DDSM_5_Processing_Functions import CS_Technique

from DDSM_2_Folders import DataCSV

from DDSM_2_Folders import NOALLBenignImages
from DDSM_2_Folders import NOBenignImages
from DDSM_2_Folders import NOBenignWCImages
from DDSM_2_Folders import NOMalignantImages

from DDSM_2_Folders import MFALLBenignImages
from DDSM_2_Folders import MFBenignImages
from DDSM_2_Folders import MFBenignWCImages
from DDSM_2_Folders import MFMalignantImages

from DDSM_2_Folders import CSALLBenignImages
from DDSM_2_Folders import CSBenignImages
from DDSM_2_Folders import CSBenignWCImages
from DDSM_2_Folders import CSMalignantImages


IB = 0  # Normal Images
IM = 1  # Tumor Images

Benign = 'Benign'
Malignant = 'Malignant'

Images_CS1_DDSM, DataFrame_CS_Data1_DDSM = CS_Technique(NOALLBenignImages, CSALLBenignImages, Benign, IB)
Images_CS2_DDSM, DataFrame_CS_Data2_DDSM = CS_Technique(NOMalignantImages, CSMalignantImages, Malignant, IM)

DataFrameCS_ALL_DDSM = pd.concat([DataFrame_CS_Data1_DDSM, DataFrame_CS_Data2_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameCS_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_CS_DDSM_Info' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameCS_ALL_DDSM)
DataFrameCS_ALL_DDSM.to_csv(dstPath)

IB = 0  # Normal Images
IBWC = 1  # Normal Images
IM = 2  # Tumor Images

Benign = 'Benign'
BenignWC = 'BenignWC'
Malignant = 'Malignant'

Images_CS1_DDSM, DataFrame_CS_Data1_DDSM = CS_Technique(NOBenignImages, CSBenignImages, Benign, IB)
Images_CS2_DDSM, DataFrame_CS_Data2_DDSM = CS_Technique(NOBenignWCImages, CSBenignWCImages, BenignWC, IBWC)
Images_CS3_DDSM, DataFrame_CS_Data3_DDSM = CS_Technique(NOMalignantImages, CSMalignantImages, Malignant, IM)

DataFrameCS_ALL_DDSM = pd.concat([DataFrame_CS_Data1_DDSM, DataFrame_CS_Data2_DDSM, DataFrame_CS_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameCS_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_CS_DDSM_Multi_Info' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameCS_ALL_DDSM)
DataFrameCS_ALL_DDSM.to_csv(dstPath)