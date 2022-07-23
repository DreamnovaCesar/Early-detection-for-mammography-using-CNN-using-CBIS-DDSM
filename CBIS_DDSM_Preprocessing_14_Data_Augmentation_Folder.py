
from CBIS_DDSM_2_General_Functions import split_folders_train_test_val

from CBIS_DDSM_1_Folders import CBIS_DDSM_NT_Images_Biclass
#from CBIS_DDSM_1_Folders import CBIS_DDSM_NT_Images_Multiclass

from CBIS_DDSM_1_Folders import CBIS_DDSM_NO_Images_Biclass
#from CBIS_DDSM_1_Folders import CBIS_DDSM_NO_Images_Multiclass


def Split_Folders_Each_Technique():

    split_folders_train_test_val(CBIS_DDSM_NT_Images_Biclass)
    #split_folders_train_test_val(Mini_MIAS_NT_Cropped_Images_Multiclass)

    split_folders_train_test_val(CBIS_DDSM_NO_Images_Biclass)
    #split_folders_train_test_val(Mini_MIAS_NO_Cropped_Images_Multiclass)

    #split_folders_train_test_val(Mini_MIAS_CLAHE_Cropped_Images_Biclass)
    #split_folders_train_test_val(Mini_MIAS_CLAHE_Cropped_Images_Multiclass)

    #split_folders_train_test_val(Mini_MIAS_HE_Cropped_Images_Biclass)
    #split_folders_train_test_val(Mini_MIAS_HE_Cropped_Images_Multiclass)

    #split_folders_train_test_val(Mini_MIAS_UM_Cropped_Images_Biclass)
    #split_folders_train_test_val(Mini_MIAS_UM_Cropped_Images_Multiclass)

    #split_folders_train_test_val(Mini_MIAS_CS_Cropped_Images_Biclass)
    #split_folders_train_test_val(Mini_MIAS_CS_Cropped_Images_Multiclass)