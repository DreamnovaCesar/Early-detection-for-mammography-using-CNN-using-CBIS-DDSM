import os
import pandas as pd

from DDSM_5_Processing_Functions import CLAHE_Technique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import GeneralDataCSV
from DDSM_2_Folders import GeneralMultiDataCSV

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import NONormalImages
from DDSM_2_Folders import NOAbnormalImages
from DDSM_2_Folders import NOAbnormalMassImages

from DDSM_2_Folders import CLAHENormalImages
from DDSM_2_Folders import CLAHEAbnormalImages
from DDSM_2_Folders import CLAHEAbnormalMassImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Clip_limit = 0.01

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Tumor

IC = 0  # Normal Images
IM = 1  # Tumor Images

Calcification = 'Calcification'
Mass = 'Mass'

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Tumor

DataFrame_Data1_DDSM = CLAHE_Technique(NOAbnormalImages, CLAHEAbnormalImages, Calcification, Clip_limit, IC)
DataFrame_Data2_DDSM = CLAHE_Technique(NOAbnormalMassImages, CLAHEAbnormalMassImages, Mass, Clip_limit, IM)

DataFrame_ALL_DDSM = pd.concat([DataFrame_Data1_DDSM, DataFrame_Data2_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrame_ALL_DDSM.shape[0] + 1)

dst = 'Biclass_Dataframe_CLAHE_Calc_Mass' + '.csv'
dstPath = os.path.join(GeneralDataCSV, dst)

print(DataFrame_ALL_DDSM)
DataFrame_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Tumor

IN = 0  # Normal Images
IM = 1  # Normal Images
IC = 2  # Tumor Images

Normal = 'Normal'
Mass = 'Mass'
Calcification = 'Calcification'

DataFrame_Data1_DDSM = CLAHE_Technique(NONormalImages, CLAHENormalImages, Normal, Clip_limit, IN)
DataFrame_Data2_DDSM = CLAHE_Technique(NOAbnormalImages, CLAHEAbnormalImages, Mass, Clip_limit, IM)
DataFrame_Data3_DDSM = CLAHE_Technique(NOAbnormalMassImages, CLAHEAbnormalMassImages, Calcification, Clip_limit, IC)

DataFrame_ALL_DDSM = pd.concat([DataFrame_Data1_DDSM, DataFrame_Data2_DDSM, DataFrame_Data3_DDSM], ignore_index = True, sort = False)

pd.set_option('display.max_rows', DataFrame_ALL_DDSM.shape[0] + 1)

dst = 'Multiclass_Dataframe_CLAHE_Calc_Mass' + '.csv'
dstPath = os.path.join(GeneralMultiDataCSV, dst)

print(DataFrame_ALL_DDSM)
DataFrame_ALL_DDSM.to_csv(dstPath)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########
