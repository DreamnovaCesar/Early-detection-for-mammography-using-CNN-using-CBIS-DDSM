
from DDSM_4_DDSM_Functions import ConvertDCM

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import Calcification_Test
from DDSM_2_Folders import Calcification_Test_Patches
from DDSM_2_Folders import Calcification_Test_Patches_Resize
from DDSM_2_Folders import Calcification_Test_Patches_Resize_Normalization

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import Calcification_Training
from DDSM_2_Folders import Calcification_Training_Patches
from DDSM_2_Folders import Calcification_Training_Patches_Resize
from DDSM_2_Folders import Calcification_Training_Patches_Resize_Normalization

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import Calcification_Test_Patches_Resize_299
from DDSM_2_Folders import Calcification_Test_Patches_Resize_Normalization_299
from DDSM_2_Folders import Calcification_Training_Patches_Resize_299
from DDSM_2_Folders import Calcification_Training_Patches_Resize_Normalization_299

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import Mass_Test
from DDSM_2_Folders import Mass_Test_Patches
from DDSM_2_Folders import Mass_Test_Patches_Resize
from DDSM_2_Folders import Mass_Test_Patches_Resize_Normalization

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import Mass_Training
from DDSM_2_Folders import Mass_Training_Patches
from DDSM_2_Folders import Mass_Training_Patches_Resize
from DDSM_2_Folders import Mass_Training_Patches_Resize_Normalization

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Calc_Name_Training = 'Cal_Training'
Calc_Name_Test = 'Cal_Test'

Mass_Name_Training = 'Mass_Training'
Mass_Name_Test = 'Mass_Test'

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Size = 224
SizeArticle = 299

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

ConvertDCM(Calcification_Test, Calc_Name_Training, Calcification_Test_Patches, Calcification_Test_Patches_Resize, Calcification_Test_Patches_Resize_Normalization, Size)
ConvertDCM(Calcification_Training, Calc_Name_Test, Calcification_Training_Patches, Calcification_Training_Patches_Resize, Calcification_Training_Patches_Resize_Normalization, Size)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

ConvertDCM(Calcification_Test, Calc_Name_Training, Calcification_Test_Patches, Calcification_Test_Patches_Resize_299, Calcification_Test_Patches_Resize_Normalization_299, SizeArticle)
ConvertDCM(Calcification_Training, Calc_Name_Test, Calcification_Training_Patches, Calcification_Training_Patches_Resize_299, Calcification_Training_Patches_Resize_Normalization_299, SizeArticle)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

ConvertDCM(Mass_Test, Mass_Name_Test, Mass_Test_Patches, Mass_Test_Patches_Resize, Mass_Test_Patches_Resize_Normalization, Size)
ConvertDCM(Mass_Training, Mass_Name_Training, Mass_Training_Patches, Mass_Training_Patches_Resize, Mass_Training_Patches_Resize_Normalization, Size)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########