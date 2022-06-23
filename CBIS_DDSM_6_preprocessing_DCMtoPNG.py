
from DDSM_4_DDSM_Functions import ConvertDCM

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Calcification_Test = 'D:\Calcification Test\CBIS-DDSM'
Calcification_Test_Patches = 'D:\Calcification Test\CBIS-DDSM Images pach'
Calcification_Test_Patches_Resize = 'D:\Calcification Test\CBIS-DDSM Images pach resize'
Calcification_Test_Patches_Resize_Normalization = 'D:\Calcification Test\CBIS-DDSM Images pach resize normalize'

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Calcification_Training = 'D:\Calcification Training\CBIS-DDSM'
Calcification_Training_Patches = 'D:\Calcification Training\CBIS-DDSM Images pach'
Calcification_Training_Patches_Resize = 'D:\Calcification Training\CBIS-DDSM Images pach resize'
Calcification_Training_Patches_Resize_Normalization = 'D:\Calcification Training\CBIS-DDSM Images pach resize normalize'

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Mass_Test = 'D:\Mass Test\CBIS-DDSM'
Mass_Test_Patches  = 'D:\Mass Test\CBIS-DDSM Mass Images patches'
Mass_Test_Patches_Resize  = 'D:\Mass Test\CBIS-DDSM Mass Images patches resize'
Mass_Test_Patches_Resize_Normalization = 'D:\Mass Test\CBIS-DDSM Mass Images patches resize normalize'

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Mass_Training = 'D:\Mass Training\CBIS-DDSM'
Mass_Training_Patches = 'D:\Mass Training\CBIS-DDSM Images pach'
Mass_Training_Patches_Resize = 'D:\Mass Training\CBIS-DDSM Mass Images patches resize'
Mass_Training_Patches_Resize_Normalization = 'D:\Mass Training\CBIS-DDSM Mass Images patches resize normalize'

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Calc_Name_Training = 'Cal_Training'
Calc_Name_Test = 'Cal_Test'

Mass_Name_Training = 'Mass_Training'
Mass_Name_Test = 'Mass_Test'

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

ConvertDCM(Calcification_Test, Calc_Name_Training, Calcification_Test_Patches, Calcification_Test_Patches_Resize, Calcification_Test_Patches_Resize_Normalization)
ConvertDCM(Calcification_Training, Calc_Name_Test, Calcification_Training_Patches, Calcification_Training_Patches_Resize, Calcification_Training_Patches_Resize_Normalization)

ConvertDCM(Mass_Test, Mass_Name_Test, Mass_Test_Patches, Mass_Test_Patches_Resize, Mass_Test_Patches_Resize_Normalization)
ConvertDCM(Mass_Training, Mass_Name_Training, Mass_Training_Patches, Mass_Training_Patches_Resize, Mass_Training_Patches_Resize_Normalization)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########