from DDSM_6_Data_Augmentation import DataAugmentation

from DDSM_2_Folders import DataCSV299

from DDSM_2_Folders import WOALLBenignImages299
from DDSM_2_Folders import WOBenignImages299
from DDSM_2_Folders import WOBenignWCImages299
from DDSM_2_Folders import WOMalignantImages299

from DDSM_2_Folders import NOALLBenignImages299
from DDSM_2_Folders import NOBenignImages299
from DDSM_2_Folders import NOBenignWCImages299
from DDSM_2_Folders import NOMalignantImages299

from DDSM_2_Folders import CLAHEALLBenignImages299
from DDSM_2_Folders import CLAHEBenignImages299
from DDSM_2_Folders import CLAHEBenignWCImages299
from DDSM_2_Folders import CLAHEMalignantImages299

Normal = 2
Tumor = 2

NNormal = 'BenignWC'
NTumor = 'Malignant'

IN = 0
IT = 1

#####

Images_BenignWC, Labels_BenignWC = DataAugmentation(UMBenignImages, NNormal, Normal, IN)

Images_Malignant, Labels_Malignant = DataAugmentation(UMMalignantImages, NTumor, Tumor, IT)
"""
#####

NOImages_BenignWC, NOLabels_BenignWC = DataAugmentation(NOBenignWCImages, NNormal, Normal, IN)

NOImages_Malignant, NOLabels_Malignant = DataAugmentation(NOMalignantImages, NTumor, Tumor, IT)

#####

CLAHEImages_BenignWC, CLAHELabels_BenignWC = DataAugmentation(CLAHEBenignWCImages, NNormal, Normal, IN)

CLAHEImages_Malignant, CLAHELabels_Malignant = DataAugmentation(CLAHEMalignantImages, NTumor, Tumor, IT)

#####

HEImages_BenignWC, HELabels_BenignWC = DataAugmentation(HEBenignWCImages, NNormal, Normal, IN)

HEImages_Malignant, HELabels_Malignant = DataAugmentation(HEMalignantImages, NTumor, Tumor, IT)

#####

UMImages_BenignWC, UMLabels_BenignWC = DataAugmentation(UMBenignWCImages, NNormal, Normal, IN)

UMImages_Malignant, UMLabels_Malignant = DataAugmentation(UMMalignantImages, NTumor, Tumor, IT)

#####

CSImages_BenignWC, CSLabels_BenignWC = DataAugmentation(CSBenignWCImages, NNormal, Normal, IN)

CSImages_Malignant, CSLabels_Malignant = DataAugmentation(CSMalignantImages, NTumor, Tumor, IT)
"""
#####

#print(len(Images_BenignWC))
#print(len(Images_Malignant))
