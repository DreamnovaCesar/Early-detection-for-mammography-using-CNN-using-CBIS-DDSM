import os
import pandas as pd

from DDSM_5_Processing_Functions import MedianFilterNoise

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import DataCSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import NOALLBenignImages
from DDSM_2_Folders import NOBenignImages
from DDSM_2_Folders import NOBenignWCImages
from DDSM_2_Folders import NOMalignantImages

from DDSM_2_Folders import MFALLBenignImages
from DDSM_2_Folders import MFBenignImages
from DDSM_2_Folders import MFBenignWCImages
from DDSM_2_Folders import MFMalignantImages

from DDSM_2_Folders import CLAHEALLBenignImages
from DDSM_2_Folders import CLAHEBenignImages
from DDSM_2_Folders import CLAHEBenignWCImages
from DDSM_2_Folders import CLAHEMalignantImages

from DDSM_2_Folders import MFCLAHEALLBenignImages
from DDSM_2_Folders import MFCLAHEBenignImages
from DDSM_2_Folders import MFCLAHEBenignWCImages
from DDSM_2_Folders import MFCLAHEMalignantImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Division = 3

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

IB = 0  # Normal Images
IM = 1  # Tumor Images

Benign = 'Benign'
Malignant = 'Malignant'

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Images_MF1_DDSM, DataFrame_MF_Data1_DDSM = MedianFilterNoise(CLAHEBenignImages, MFCLAHEBenignImages, Benign, Division, IB)
Images_MF3_DDSM, DataFrame_MF_Data3_DDSM = MedianFilterNoise(CLAHEMalignantImages, MFCLAHEMalignantImages, Malignant, Division, IM)

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

Images_MF1_DDSM, DataFrame_MF_Data1_DDSM = MedianFilterNoise(CLAHEBenignImages, MFCLAHEBenignImages, Benign, Division, IB)
Images_MF2_DDSM, DataFrame_MF_Data2_DDSM = MedianFilterNoise(CLAHEBenignWCImages, MFCLAHEBenignWCImages, BenignWC, Division, IBWC)
#Images_MF3_DDSM, DataFrame_MF_Data3_DDSM = MedianFilterNoise(CLAHEMalignantImages, MFCLAHEMalignantImages, Malignant, Division, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_MF_Data1_DDSM, DataFrame_MF_Data2_DDSM, DataFrame_MF_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_MedianFilter_DDSM_Multi_Info' + '.csv'
dstPath = os.path.join(DataCSV, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)