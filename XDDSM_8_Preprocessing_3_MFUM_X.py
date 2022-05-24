import os
import pandas as pd

from DDSM_5_Processing_Functions import MedianFilterNoise

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

Division = 3

IB = 0  # Normal Images
IM = 1  # Tumor Images

Benign = 'Benign'
Malignant = 'Malignant'

Images_MF1_DDSM, DataFrame_MF_Data1_DDSM = MedianFilterNoise(UMBenignImages, MFUMBenignImages, Benign, Division, IB)
Images_MF3_DDSM, DataFrame_MF_Data3_DDSM = MedianFilterNoise(UMMalignantImages, MFUMMalignantImages, Malignant, Division, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_MF_Data1_DDSM, DataFrame_MF_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_MedianFilter_DDSM_Info' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)

IB = 0  # Normal Images
IBWC = 1  # Normal Images
IM = 2  # Tumor Images

Benign = 'Benign'
BenignWC = 'BenignWC'
Malignant = 'Malignant'

#Images_MF1_DDSM, DataFrame_MF_Data1_DDSM = MedianFilterNoise(NOBenignImages, MFBenignImages, Benign, Division, IB)
Images_MF2_DDSM, DataFrame_MF_Data2_DDSM = MedianFilterNoise(UMBenignWCImages, MFUMBenignWCImages, BenignWC, Division, IBWC)
#Images_MF3_DDSM, DataFrame_MF_Data3_DDSM = MedianFilterNoise(NOMalignantImages, CLAHEMalignantImages, Malignant, Division, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_MF_Data1_DDSM, DataFrame_MF_Data2_DDSM, DataFrame_MF_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_MedianFilter_DDSM_Multi_Info' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)