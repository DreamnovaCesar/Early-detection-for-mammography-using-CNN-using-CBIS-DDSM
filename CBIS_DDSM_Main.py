import numpy as np

from CBIS_DDSM_1_Folders import Biclass_Data_CSV
from CBIS_DDSM_1_Folders import Biclass_Data_Models
from CBIS_DDSM_1_Folders import Biclass_Data_Models_Esp

from CBIS_DDSM_1_Folders import Multiclass_Data_CSV
from CBIS_DDSM_1_Folders import Multiclass_Data_Models
from CBIS_DDSM_1_Folders import Multiclass_Data_Models_Esp

from CBIS_DDSM_2_General_Functions import concat_dataframe

from CBIS_DDSM_preprocessing_1_Change_Format import preprocessing_ChangeFormat

from CBIS_DDSM_preprocessing_2_Split_Calc_Mass import preprocessing_SplitCalcMass

from CBIS_DDSM_preprocessing_3_Select_technique import preprocessing_technique_Biclass
from CBIS_DDSM_preprocessing_3_Select_technique import preprocessing_technique_Multiclass

from CBIS_DDSM_preprocessing_3_Select_technique import preprocessing_technique_Biclass
from CBIS_DDSM_preprocessing_3_Select_technique import preprocessing_technique_Multiclass

from CBIS_DDSM_Preprocessing_10_Data_Augmentation import preprocessing_DataAugmentation_Biclass_ML
from CBIS_DDSM_Preprocessing_10_Multi_Data_Augmentation import preprocessing_DataAugmentation_Multiclass_ML

from CBIS_DDSM_Preprocessing_10_Data_Augmentation import preprocessing_DataAugmentation_Biclass_CNN
from CBIS_DDSM_Preprocessing_10_Multi_Data_Augmentation import preprocessing_DataAugmentation_Multiclass_CNN

from CBIS_DDSM_ML_FeaturesExtraction import Testing_ML_Models_Biclass_FOF
from CBIS_DDSM_ML_FeaturesExtraction import Testing_ML_Models_Multiclass_FOF
from CBIS_DDSM_ML_FeaturesExtraction import Testing_ML_Models_Biclass_GLCM
from CBIS_DDSM_ML_FeaturesExtraction import Testing_ML_Models_Multiclass_GLCM

from CBIS_DDSM_Preprocessing_13_CNN_Models_Folder import Testing_CNN_Models_Biclass_From_Folder
from CBIS_DDSM_Preprocessing_13_CNN_Models_Folder import Testing_CNN_Models_Multiclass_From_Folder

from CBIS_DDSM_Preprocessing_14_Data_Augmentation_Folder import Split_Folders_Each_Technique

from CBIS_DDSM_1_Folders import Normal_NT_Images
from CBIS_DDSM_1_Folders import Normal_NO_Images
from CBIS_DDSM_1_Folders import Normal_CLAHE_Images
from CBIS_DDSM_1_Folders import Normal_HE_Images
from CBIS_DDSM_1_Folders import Normal_UM_Images
from CBIS_DDSM_1_Folders import Normal_CS_Images

from CBIS_DDSM_1_Folders import Calc_NT_All_Benign_Images
from CBIS_DDSM_1_Folders import Calc_NT_Benign_Images
from CBIS_DDSM_1_Folders import Calc_NT_Benign_WC_Images
from CBIS_DDSM_1_Folders import Calc_NT_Malignant_Images
from CBIS_DDSM_1_Folders import Calc_NT_Abnormal_Images

from CBIS_DDSM_1_Folders import Mass_NT_All_Benign_Images
from CBIS_DDSM_1_Folders import Mass_NT_Benign_Images
from CBIS_DDSM_1_Folders import Mass_NT_Benign_WC_Images
from CBIS_DDSM_1_Folders import Mass_NT_Malignant_Images
from CBIS_DDSM_1_Folders import Mass_NT_Abnormal_Images

from CBIS_DDSM_1_Folders import Calc_NO_All_Benign_Images
from CBIS_DDSM_1_Folders import Calc_NO_Benign_Images
from CBIS_DDSM_1_Folders import Calc_NO_Benign_WC_Images
from CBIS_DDSM_1_Folders import Calc_NO_Malignant_Images
from CBIS_DDSM_1_Folders import Calc_NO_Abnormal_Images

from CBIS_DDSM_1_Folders import Mass_NO_All_Benign_Images
from CBIS_DDSM_1_Folders import Mass_NO_Benign_Images
from CBIS_DDSM_1_Folders import Mass_NO_Benign_WC_Images
from CBIS_DDSM_1_Folders import Mass_NO_Malignant_Images
from CBIS_DDSM_1_Folders import Mass_NO_Abnormal_Images

from CBIS_DDSM_1_Folders import Calc_CLAHE_All_Benign_Images
from CBIS_DDSM_1_Folders import Calc_CLAHE_Benign_Images
from CBIS_DDSM_1_Folders import Calc_CLAHE_Benign_WC_Images
from CBIS_DDSM_1_Folders import Calc_CLAHE_Malignant_Images
from CBIS_DDSM_1_Folders import Calc_CLAHE_Abnormal_Images

from CBIS_DDSM_1_Folders import Mass_CLAHE_All_Benign_Images
from CBIS_DDSM_1_Folders import Mass_CLAHE_Benign_Images
from CBIS_DDSM_1_Folders import Mass_CLAHE_Benign_WC_Images
from CBIS_DDSM_1_Folders import Mass_CLAHE_Malignant_Images
from CBIS_DDSM_1_Folders import Mass_CLAHE_Abnormal_Images

from CBIS_DDSM_1_Folders import Calc_HE_All_Benign_Images
from CBIS_DDSM_1_Folders import Calc_HE_Benign_Images
from CBIS_DDSM_1_Folders import Calc_HE_Benign_WC_Images
from CBIS_DDSM_1_Folders import Calc_HE_Malignant_Images
from CBIS_DDSM_1_Folders import Calc_HE_Abnormal_Images

from CBIS_DDSM_1_Folders import Mass_HE_All_Benign_Images
from CBIS_DDSM_1_Folders import Mass_HE_Benign_Images
from CBIS_DDSM_1_Folders import Mass_HE_Benign_WC_Images
from CBIS_DDSM_1_Folders import Mass_HE_Malignant_Images
from CBIS_DDSM_1_Folders import Mass_HE_Abnormal_Images

from CBIS_DDSM_1_Folders import Calc_UM_All_Benign_Images
from CBIS_DDSM_1_Folders import Calc_UM_Benign_Images
from CBIS_DDSM_1_Folders import Calc_UM_Benign_WC_Images
from CBIS_DDSM_1_Folders import Calc_UM_Malignant_Images
from CBIS_DDSM_1_Folders import Calc_UM_Abnormal_Images

from CBIS_DDSM_1_Folders import Mass_UM_All_Benign_Images
from CBIS_DDSM_1_Folders import Mass_UM_Benign_Images
from CBIS_DDSM_1_Folders import Mass_UM_Benign_WC_Images
from CBIS_DDSM_1_Folders import Mass_UM_Malignant_Images
from CBIS_DDSM_1_Folders import Mass_UM_Abnormal_Images

from CBIS_DDSM_1_Folders import Calc_CS_All_Benign_Images
from CBIS_DDSM_1_Folders import Calc_CS_Benign_Images
from CBIS_DDSM_1_Folders import Calc_CS_Benign_WC_Images
from CBIS_DDSM_1_Folders import Calc_CS_Malignant_Images
from CBIS_DDSM_1_Folders import Calc_CS_Abnormal_Images

from CBIS_DDSM_1_Folders import Mass_CS_All_Benign_Images
from CBIS_DDSM_1_Folders import Mass_CS_Benign_Images
from CBIS_DDSM_1_Folders import Mass_CS_Benign_WC_Images
from CBIS_DDSM_1_Folders import Mass_CS_Malignant_Images
from CBIS_DDSM_1_Folders import Mass_CS_Abnormal_Images

from CBIS_DDSM_1_Folders import CBIS_DDSM_NT_Images_Biclass
from CBIS_DDSM_1_Folders import CBIS_DDSM_NT_Images_Multiclass

from CBIS_DDSM_1_Folders import CBIS_DDSM_NO_Images_Biclass
from CBIS_DDSM_1_Folders import CBIS_DDSM_NO_Images_Multiclass

from CBIS_DDSM_1_Folders import CBIS_DDSM_CLAHE_Images_Biclass
from CBIS_DDSM_1_Folders import CBIS_DDSM_CLAHE_Images_Multiclass

from CBIS_DDSM_1_Folders import CBIS_DDSM_HE_Images_Biclass
from CBIS_DDSM_1_Folders import CBIS_DDSM_HE_Images_Multiclass

from CBIS_DDSM_1_Folders import CBIS_DDSM_UM_Images_Biclass
from CBIS_DDSM_1_Folders import CBIS_DDSM_UM_Images_Multiclass

from CBIS_DDSM_1_Folders import CBIS_DDSM_CS_Images_Biclass
from CBIS_DDSM_1_Folders import CBIS_DDSM_CS_Images_Multiclass

from CBIS_DDSM_8_CNN_Architectures import MobileNet_pretrained
from CBIS_DDSM_8_CNN_Architectures import MobileNetV3Small_pretrained
from CBIS_DDSM_8_CNN_Architectures import ResNet50_pretrained
from CBIS_DDSM_8_CNN_Architectures import ResNet152_pretrained

from CBIS_DDSM_ML_Functions import SVM
from CBIS_DDSM_ML_Functions import Multi_SVM
from CBIS_DDSM_ML_Functions import MLP
from CBIS_DDSM_ML_Functions import KNN
from CBIS_DDSM_ML_Functions import RF
from CBIS_DDSM_ML_Functions import DT
from CBIS_DDSM_ML_Functions import GBC


def main():

    Model_CNN = (ResNet152_pretrained, ResNet50_pretrained)

    Model_ML_Biclass = (SVM, MLP, KNN, RF, DT, GBC)
    Model_ML_Multiclass = (Multi_SVM, MLP, KNN, RF, DT, GBC)

    #preprocessing_ChangeFormat()
    #preprocessing_SplitCalcMass()


    """
    preprocessing_technique_Biclass('CLAHE', Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, Calc_CLAHE_Abnormal_Images, Mass_CLAHE_Abnormal_Images)
    preprocessing_technique_Biclass('HE', Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, Calc_HE_Abnormal_Images, Mass_HE_Abnormal_Images)
    preprocessing_technique_Biclass('UM', Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, Calc_UM_Abnormal_Images, Mass_UM_Abnormal_Images)
    preprocessing_technique_Biclass('CS', Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, Calc_CS_Abnormal_Images, Mass_CS_Abnormal_Images)
    
    """

    """

    preprocessing_technique_Multiclass( 'CLAHE', Normal_NO_Images, Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, 
                                            Normal_CLAHE_Images, Calc_CLAHE_Abnormal_Images, Mass_CLAHE_Abnormal_Images)

    preprocessing_technique_Multiclass( 'HE', Normal_NO_Images, Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, 
                                            Normal_HE_Images, Calc_HE_Abnormal_Images, Mass_HE_Abnormal_Images)

    preprocessing_technique_Multiclass( 'UM', Normal_NO_Images, Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, 
                                            Normal_UM_Images, Calc_UM_Abnormal_Images, Mass_UM_Abnormal_Images)

    preprocessing_technique_Multiclass( 'CS', Normal_NO_Images, Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, 
                                            Normal_CS_Images, Calc_CS_Abnormal_Images, Mass_CS_Abnormal_Images)
    
    """

    """

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_NT_Abnormal_Images, Mass_NT_Abnormal_Images, CBIS_DDSM_NT_Images_Biclass)
    Dataframe_NT_FOF = Testing_ML_Models_Biclass_FOF(Model_ML_Biclass, 'NT', Images, Labels)
    Dataframe_NT_GLCM = Testing_ML_Models_Biclass_GLCM(Model_ML_Biclass, 'NT', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, CBIS_DDSM_NO_Images_Biclass)
    Dataframe_NO_FOF = Testing_ML_Models_Biclass_FOF(Model_ML_Biclass, 'NO', Images, Labels)
    Dataframe_NO_GLCM = Testing_ML_Models_Biclass_GLCM(Model_ML_Biclass, 'NO', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_CLAHE_Abnormal_Images, Mass_CLAHE_Abnormal_Images, CBIS_DDSM_CLAHE_Images_Biclass)
    Dataframe_CLAHE_FOF = Testing_ML_Models_Biclass_FOF(Model_ML_Biclass, 'CLAHE', Images, Labels)
    Dataframe_CLAHE_GLCM = Testing_ML_Models_Biclass_GLCM(Model_ML_Biclass, 'CLAHE', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_HE_Abnormal_Images, Mass_HE_Abnormal_Images, CBIS_DDSM_HE_Images_Biclass)
    Dataframe_HE_FOF = Testing_ML_Models_Biclass_FOF(Model_ML_Biclass, 'HE', Images, Labels)
    Dataframe_HE_GLCM = Testing_ML_Models_Biclass_GLCM(Model_ML_Biclass, 'HE', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_UM_Abnormal_Images, Mass_UM_Abnormal_Images, CBIS_DDSM_UM_Images_Biclass)
    Dataframe_UM_FOF = Testing_ML_Models_Biclass_FOF(Model_ML_Biclass, 'UM', Images, Labels)
    Dataframe_UM_GLCM = Testing_ML_Models_Biclass_GLCM(Model_ML_Biclass, 'UM', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_CS_Abnormal_Images, Mass_CS_Abnormal_Images, CBIS_DDSM_CS_Images_Biclass)
    Dataframe_CS_FOF = Testing_ML_Models_Biclass_FOF(Model_ML_Biclass, 'CS', Images, Labels)
    Dataframe_CS_GLCM = Testing_ML_Models_Biclass_GLCM(Model_ML_Biclass, 'CS', Images, Labels)

    concat_dataframe(Dataframe_NT_FOF, Dataframe_NO_FOF, Dataframe_CLAHE_FOF, Dataframe_HE_FOF, Dataframe_UM_FOF, Dataframe_CS_FOF, Folder = Biclass_Data_CSV, Class = 'Biclass', Technique = 'FOF_All_Techniques', SaveCSV = True)
    concat_dataframe(Dataframe_NT_GLCM, Dataframe_NO_GLCM, Dataframe_CLAHE_GLCM, Dataframe_HE_GLCM, Dataframe_UM_GLCM, Dataframe_CS_GLCM, Folder = Biclass_Data_CSV, Class = 'Biclass', Technique = 'GLCM_All_Techniques', SaveCSV = True)

    """

    """

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_NT_Images, Calc_NT_Abnormal_Images, Mass_NT_Abnormal_Images, CBIS_DDSM_NT_Images_Multiclass)
    Dataframe_NT_FOF = Testing_ML_Models_Multiclass_FOF(Model_ML_Multiclass, 'NT', Images, Labels)
    Dataframe_NT_GLCM = Testing_ML_Models_Multiclass_GLCM(Model_ML_Multiclass, 'NT', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_NO_Images, Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, CBIS_DDSM_NO_Images_Multiclass)
    Dataframe_NO_FOF = Testing_ML_Models_Multiclass_FOF(Model_ML_Multiclass, 'NO', Images, Labels)
    Dataframe_NO_GLCM = Testing_ML_Models_Multiclass_GLCM(Model_ML_Multiclass, 'NO', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_CLAHE_Images, Calc_CLAHE_Abnormal_Images, Mass_CLAHE_Abnormal_Images, CBIS_DDSM_CLAHE_Images_Multiclass)
    Dataframe_CLAHE_FOF = Testing_ML_Models_Multiclass_FOF(Model_ML_Multiclass, 'CLAHE', Images, Labels)
    Dataframe_CLAHE_GLCM = Testing_ML_Models_Multiclass_GLCM(Model_ML_Multiclass, 'CLAHE', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_HE_Images, Calc_HE_Abnormal_Images, Mass_HE_Abnormal_Images, CBIS_DDSM_HE_Images_Multiclass)
    Dataframe_HE_FOF = Testing_ML_Models_Multiclass_FOF(Model_ML_Multiclass, 'HE', Images, Labels)
    Dataframe_HE_GLCM = Testing_ML_Models_Multiclass_GLCM(Model_ML_Multiclass, 'HE', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_UM_Images, Calc_UM_Abnormal_Images, Mass_UM_Abnormal_Images, CBIS_DDSM_UM_Images_Multiclass)
    Dataframe_UM_FOF = Testing_ML_Models_Multiclass_FOF(Model_ML_Multiclass, 'UM', Images, Labels)
    Dataframe_UM_GLCM = Testing_ML_Models_Multiclass_GLCM(Model_ML_Multiclass, 'UM', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_CS_Images, Calc_CS_Abnormal_Images, Mass_CS_Abnormal_Images, CBIS_DDSM_CS_Images_Multiclass)
    Dataframe_CS_FOF = Testing_ML_Models_Multiclass_FOF(Model_ML_Multiclass, 'CS', Images, Labels)
    Dataframe_CS_GLCM = Testing_ML_Models_Multiclass_GLCM(Model_ML_Multiclass, 'CS', Images, Labels)

    concat_dataframe(Dataframe_NT_FOF, Dataframe_NO_FOF, Dataframe_CLAHE_FOF, Dataframe_HE_FOF, Dataframe_UM_FOF, Dataframe_CS_FOF, Folder = Multiclass_Data_CSV, Class = 'Multiclass', Technique = 'FOF_All_Techniques', SaveCSV = True)
    concat_dataframe(Dataframe_NT_GLCM, Dataframe_NO_GLCM, Dataframe_CLAHE_GLCM, Dataframe_HE_GLCM, Dataframe_UM_GLCM, Dataframe_CS_GLCM, Folder = Multiclass_Data_CSV, Class = 'Multiclass', Technique = 'GLCM_All_Techniques', SaveCSV = True)

    """

    # * CNN

    #Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, CBIS_DDSM_NO_Images_Biclass)

    """

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_NT_Abnormal_Images, Mass_NT_Abnormal_Images, CBIS_DDSM_NT_Images_Biclass)
    Dataframe_final_NT = Testing_CNN_Models_Biclass(Model_CNN, 'NT', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, CBIS_DDSM_NO_Images_Biclass)
    Dataframe_final_NO = Testing_CNN_Models_Biclass(Model_CNN, 'NO', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_CLAHE_Abnormal_Images, Mass_CLAHE_Abnormal_Images, CBIS_DDSM_CLAHE_Images_Biclass)
    Dataframe_final_CLAHE = Testing_CNN_Models_Biclass(Model_CNN, 'CLAHE', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_HE_Abnormal_Images, Mass_HE_Abnormal_Images, CBIS_DDSM_HE_Images_Biclass)
    Dataframe_final_HE = Testing_CNN_Models_Biclass(Model_CNN, 'HE', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_HE_Abnormal_Images, Mass_HE_Abnormal_Images, CBIS_DDSM_HE_Images_Biclass)
    Dataframe_final_UM = Testing_CNN_Models_Biclass(Model_CNN, 'UM', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Biclass_ML(Calc_HE_Abnormal_Images, Mass_HE_Abnormal_Images, CBIS_DDSM_HE_Images_Biclass)
    Dataframe_final_CS = Testing_CNN_Models_Biclass(Model_CNN, 'CS', Images, Labels)

    concat_dataframe(Dataframe_final_NT, Dataframe_final_NO, Dataframe_final_CLAHE, Dataframe_final_HE, Dataframe_final_UM, Dataframe_final_CS, Folder = Biclass_Data_CSV, Class = 'Biclass', Technique = 'All techniques')

    """

    """

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_NT_Images, Calc_NT_Abnormal_Images, Mass_NT_Abnormal_Images, CBIS_DDSM_NT_Images_Multiclass)
    Dataframe_final_NT = Testing_CNN_Models_Multiclass(Model_CNN_R, 'NT', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_NO_Images, Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, CBIS_DDSM_NO_Images_Multiclass)
    Dataframe_final_NO = Testing_CNN_Models_Multiclass(Model_CNN_R, 'NO', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_CLAHE_Images, Calc_CLAHE_Abnormal_Images, Mass_CLAHE_Abnormal_Images, CBIS_DDSM_CLAHE_Images_Multiclass)
    Dataframe_final_CLAHE = Testing_CNN_Models_Multiclass(Model_CNN_R, 'CLAHE', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_HE_Images, Calc_HE_Abnormal_Images, Mass_HE_Abnormal_Images, CBIS_DDSM_HE_Images_Multiclass)
    Dataframe_final_HE = Testing_CNN_Models_Multiclass(Model_CNN_R, 'HE', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_UM_Images, Calc_UM_Abnormal_Images, Mass_UM_Abnormal_Images, CBIS_DDSM_UM_Images_Multiclass)
    Dataframe_final_UM = Testing_CNN_Models_Multiclass(Model_CNN_R, 'UM', Images, Labels)

    Images, Labels = preprocessing_DataAugmentation_Multiclass_ML(Normal_CS_Images, Calc_CS_Abnormal_Images, Mass_CS_Abnormal_Images, CBIS_DDSM_CS_Images_Multiclass)
    Dataframe_final_CS = Testing_CNN_Models_Multiclass(Model_CNN_R, 'CS', Images, Labels)

    concat_dataframe(Dataframe_final_NT, Dataframe_final_NO, Dataframe_final_CLAHE, Dataframe_final_HE, Dataframe_final_UM, Dataframe_final_CS, Folder = Multiclass_Data_CSV, Class = 'Multiclass', Technique = 'All techniques')
    
    """ 
    #d, s = preprocessing_DataAugmentation_Biclass_CNN(Calc_NO_Abnormal_Images, Mass_NO_Abnormal_Images, CBIS_DDSM_NO_Images_Biclass)
    
    Split_Folders_Each_Technique()

    #Testing_CNN_Models_Biclass_From_Folder(Model_CNN, CBIS_DDSM_NT_Images_Biclass + '_Split', 'NT')
    Testing_CNN_Models_Biclass_From_Folder(Model_CNN, CBIS_DDSM_NO_Images_Biclass + '_Split', 'NO')

if __name__ == "__main__":
    main()
