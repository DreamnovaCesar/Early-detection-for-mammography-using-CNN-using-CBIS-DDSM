import os
import pandas as pd

from DDSM_5_Processing_Functions import CLAHE_Technique

from DDSM_2_Folders import DataCSV299

from DDSM_2_Folders import NOALLBenignImages299
from DDSM_2_Folders import NOBenignImages299
from DDSM_2_Folders import NOBenignWCImages299
from DDSM_2_Folders import NOMalignantImages299
from DDSM_2_Folders import NOAbnormalImages299

from DDSM_2_Folders import CLAHEALLBenignImages299
from DDSM_2_Folders import CLAHEBenignImages299
from DDSM_2_Folders import CLAHEBenignWCImages299
from DDSM_2_Folders import CLAHEMalignantImages299
from DDSM_2_Folders import CLAHEAbnormalImages299

Clip_limit = 0.01

IB = 0  # Normal Images
IM = 1  # Tumor Images

Benign = 'Benign'
Malignant = 'Malignant'

Images_CLAHE1_DDSM, DataFrame_CLAHE_Data1_DDSM = CLAHE_Technique(NOALLBenignImages299, CLAHEALLBenignImages299, Benign, Clip_limit, IB)
Images_CLAHE3_DDSM, DataFrame_CLAHE_Data3_DDSM = CLAHE_Technique(NOMalignantImages299, CLAHEMalignantImages299, Malignant, Clip_limit, IM)

DataFrameCLAHE_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameCLAHE_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_CLAHE_DDSM_Bi_299' + '.csv'
dstPath = os.path.join(DataCSV299, dst)

print(DataFrameCLAHE_ALL_DDSM)
DataFrameCLAHE_ALL_DDSM.to_csv(dstPath)

IB = 0  # Normal Images
IBWC = 1  # Normal Images
IM = 2  # Tumor Images

Benign = 'Benign'
BenignWC = 'BenignWC'
Malignant = 'Malignant'

Images_CLAHE1_DDSM, DataFrame_CLAHE_Data1_DDSM = CLAHE_Technique(NOBenignImages299, CLAHEBenignImages299, Benign, Clip_limit, IB)
Images_CLAHE2_DDSM, DataFrame_CLAHE_Data2_DDSM = CLAHE_Technique(NOBenignWCImages299, CLAHEBenignWCImages299, BenignWC, Clip_limit, IBWC)
Images_CLAHE4_DDSM, DataFrame_CLAHE_Data4_DDSM = CLAHE_Technique(NOAbnormalImages299, CLAHEAbnormalImages299, Malignant, Clip_limit, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data2_DDSM, DataFrame_CLAHE_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_CLAHE_DDSM_Multi_All_299' + '.csv'
dstPath = os.path.join(DataCSV299, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)

IB = 0  # Normal Images
IM = 1  # Normal Images

Images_CLAHE1_DDSM, DataFrame_CLAHE_Data1_DDSM = CLAHE_Technique(NOBenignImages299, CLAHEBenignImages299, Benign, Clip_limit, IB)
Images_CLAHE2_DDSM, DataFrame_CLAHE_Data2_DDSM = CLAHE_Technique(NOAbnormalImages299, CLAHEAbnormalImages299, Malignant, Clip_limit, IM)

DataFrameMF_ALL_DDSM = pd.concat([DataFrame_CLAHE_Data1_DDSM, DataFrame_CLAHE_Data2_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrameMF_ALL_DDSM.shape[0] + 1)

dst = 'DataFrame_CLAHE_DDSM_Multi_Abnormal_299' + '.csv'
dstPath = os.path.join(DataCSV299, dst)

print(DataFrameMF_ALL_DDSM)
DataFrameMF_ALL_DDSM.to_csv(dstPath)