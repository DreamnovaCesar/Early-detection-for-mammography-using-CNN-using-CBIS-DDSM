import sys
import numpy as np

from DDSM_4_DDSM_Functions import CSV_Calc_DDSM

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import Calcification_Test_Patches_Resize_299, Calcification_Test_Patches_Resize_Normalization_299, Calcification_Training_Patches_Resize, Calcification_Training_Patches_Resize_299, Calcification_Training_Patches_Resize_Normalization_299
from DDSM_2_Folders import Calcification_Test_Patches_Resize

from DDSM_2_Folders import Calcification_Training_Patches_Resize_Normalization
from DDSM_2_Folders import Calcification_Test_Patches_Resize_Normalization

from DDSM_2_Folders import Mass_Training_Patches_Resize
from DDSM_2_Folders import Mass_Test_Patches_Resize

from DDSM_2_Folders import Mass_Training_Patches_Resize_Normalization
from DDSM_2_Folders import Mass_Test_Patches_Resize_Normalization

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Calcification

from DDSM_2_Folders import WOALLBenignImages
from DDSM_2_Folders import WOBenignImages
from DDSM_2_Folders import WOBenignWCImages
from DDSM_2_Folders import WOMalignantImages
from DDSM_2_Folders import WOAbnormalImages

from DDSM_2_Folders import NOALLBenignImages
from DDSM_2_Folders import NOBenignImages
from DDSM_2_Folders import NOBenignWCImages
from DDSM_2_Folders import NOMalignantImages
from DDSM_2_Folders import NOAbnormalImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Calcification

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Mass

from DDSM_2_Folders import WOALLBenignMassImages
from DDSM_2_Folders import WOBenignMassImages
from DDSM_2_Folders import WOBenignWCMassImages
from DDSM_2_Folders import WOMalignantMassImages
from DDSM_2_Folders import WOAbnormalMassImages

from DDSM_2_Folders import NOALLBenignMassImages
from DDSM_2_Folders import NOBenignMassImages
from DDSM_2_Folders import NOBenignWCMassImages
from DDSM_2_Folders import NOMalignantMassImages
from DDSM_2_Folders import NOAbnormalMassImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Mass

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Calcification 299

from DDSM_2_Folders import WOALLBenignImages299
from DDSM_2_Folders import WOBenignImages299
from DDSM_2_Folders import WOBenignWCImages299
from DDSM_2_Folders import WOMalignantImages299
from DDSM_2_Folders import WOAbnormalImages299

from DDSM_2_Folders import NOALLBenignImages299
from DDSM_2_Folders import NOBenignImages299
from DDSM_2_Folders import NOBenignWCImages299
from DDSM_2_Folders import NOMalignantImages299
from DDSM_2_Folders import NOAbnormalImages299

########## ########## ########## ########## ########## ########## ########## ########## ########## ########## Calcification 299

from DDSM_4_DDSM_Functions import TransformedLabelDDSM
from DDSM_4_DDSM_Functions import SplitDataDDSMMod
from DDSM_4_DDSM_Functions import SplitDataDDSM

np.set_printoptions(threshold = sys.maxsize)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Calc_Description_Test = "D:\DDSM\Calc Mammography\calc_case_description_test_set.csv"
Calc_Description_Training = "D:\DDSM\Calc Mammography\calc_case_description_train_set.csv"

Mass_Description_Test = "D:\DDSM\Mass Mammography\mass_case_description_test_set.csv"
Mass_Description_Training = "D:\DDSM\Mass Mammography\mass_case_description_train_set.csv"

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Pathology = 9

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Dataset_Calc_Pathology_Test = CSV_Calc_DDSM(Calc_Description_Test, Pathology, 1)
Dataset_Calc_Pathology_Training = CSV_Calc_DDSM(Calc_Description_Training, Pathology, 1)

Dataset_Mass_Pathology_Test = CSV_Calc_DDSM(Mass_Description_Test, Pathology, 2)
Dataset_Mass_Pathology_Training = CSV_Calc_DDSM(Mass_Description_Training, Pathology, 2)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Images_DDSM, Dataset_DDSM = SplitDataDDSM(Calcification_Test_Patches_Resize, Calcification_Training_Patches_Resize, WOALLBenignImages, WOBenignImages, WOBenignWCImages, WOMalignantImages, WOAbnormalImages, Dataset_Calc_Pathology_Test, Dataset_Calc_Pathology_Training)
Images_DDSM, Dataset_DDSM = SplitDataDDSM(Calcification_Test_Patches_Resize_Normalization, Calcification_Training_Patches_Resize_Normalization, NOALLBenignImages, NOBenignImages, NOBenignWCImages, NOMalignantImages, NOAbnormalImages, Dataset_Calc_Pathology_Test, Dataset_Calc_Pathology_Training)

Images_DDSM, Dataset_DDSM = SplitDataDDSM(Calcification_Test_Patches_Resize_299, Calcification_Training_Patches_Resize_299, WOALLBenignImages299, WOBenignImages299, WOBenignWCImages299, WOMalignantImages299, WOAbnormalImages299, Dataset_Calc_Pathology_Test, Dataset_Calc_Pathology_Training)
Images_DDSM, Dataset_DDSM = SplitDataDDSM(Calcification_Test_Patches_Resize_Normalization_299, Calcification_Training_Patches_Resize_Normalization_299, NOALLBenignImages299, NOBenignImages299, NOBenignWCImages299, NOMalignantImages299, NOAbnormalImages299, Dataset_Calc_Pathology_Test, Dataset_Calc_Pathology_Training)

Images_DDSM, Dataset_DDSM = SplitDataDDSM(Mass_Test_Patches_Resize, Mass_Training_Patches_Resize, WOALLBenignMassImages, WOBenignMassImages, WOBenignWCMassImages, WOMalignantMassImages, WOAbnormalMassImages, Dataset_Mass_Pathology_Test, Dataset_Mass_Pathology_Training)
Images_DDSM, Dataset_DDSM = SplitDataDDSM(Mass_Test_Patches_Resize_Normalization, Mass_Training_Patches_Resize_Normalization, NOALLBenignMassImages, NOBenignMassImages, NOBenignWCMassImages, NOMalignantMassImages, NOAbnormalMassImages, Dataset_Mass_Pathology_Test, Dataset_Mass_Pathology_Training)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########