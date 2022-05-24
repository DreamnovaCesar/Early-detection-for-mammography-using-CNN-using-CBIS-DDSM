import os
import pandas as pd

from DDSM_5_Processing_Functions import UM_Technique

from DDSM_2_Folders import DataCSV

from DDSM_2_Folders import NOALLBenignImages
from DDSM_2_Folders import NOBenignImages
from DDSM_2_Folders import NOBenignWCImages
from DDSM_2_Folders import NOMalignantImages

from DDSM_2_Folders import MFALLBenignImages
from DDSM_2_Folders import MFBenignImages
from DDSM_2_Folders import MFBenignWCImages
from DDSM_2_Folders import MFMalignantImages

from DDSM_2_Folders import UMALLBenignImages
from DDSM_2_Folders import UMBenignImages
from DDSM_2_Folders import UMBenignWCImages
from DDSM_2_Folders import UMMalignantImages

from DDSM_2_Folders import MFUMALLBenignImages
from DDSM_2_Folders import MFUMBenignImages
from DDSM_2_Folders import MFUMBenignWCImages
from DDSM_2_Folders import MFUMMalignantImages

Radius = 1
Amount = 2
IB = 0  # Normal Images
IM = 1  # Tumor Images

Benign = 'Benign'
Malignant = 'Malignant'

Images_UM1_DDSM, DataFrame_UM_Data1_DDSM = UM_Technique(NOALLBenignImages, UMALLBenignImages, Benign, Radius, Amount, IB)
Images_UM2_DDSM, DataFrame_UM_Data2_DDSM = UM_Technique(NOMalignantImages, UMMalignantImages, Malignant, Radius, Amount, IM)

DataFrameUM_ALL_DDSM = pd.concat([DataFrame_UM_Data1_DDSM, DataFrame_UM_Data2_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameUM_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_UM_DDSM_Info' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameUM_ALL_DDSM)
DataFrameUM_ALL_DDSM.to_csv(dstPath)

IB = 0  # Normal Images
IBWC = 1  # Normal Images
IM = 2  # Tumor Images

Benign = 'Benign'
BenignWC = 'BenignWC'
Malignant = 'Malignant'

Images_UM1_DDSM, DataFrame_UM_Data1_DDSM = UM_Technique(NOBenignImages, UMBenignImages, Benign, Radius, Amount, IB)
Images_UM3_DDSM, DataFrame_UM_Data3_DDSM = UM_Technique(NOBenignWCImages, UMBenignWCImages, BenignWC, Radius, Amount, IBWC)
Images_UM3_DDSM, DataFrame_UM_Data3_DDSM = UM_Technique(NOMalignantImages, UMMalignantImages, Malignant, Radius, Amount, IM)

DataFrameUM_ALL_DDSM = pd.concat([DataFrame_UM_Data1_DDSM, DataFrame_UM_Data2_DDSM, DataFrame_UM_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameUM_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_UM_DDSM_Multi_Info' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameUM_ALL_DDSM)
DataFrameUM_ALL_DDSM.to_csv(dstPath)