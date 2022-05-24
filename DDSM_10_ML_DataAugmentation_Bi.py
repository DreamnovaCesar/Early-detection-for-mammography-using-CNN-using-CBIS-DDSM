import os
import cv2
import numpy as np
import pandas as pd

from DDSM_2_Folders import DataCSV

from DDSM_2_Folders import WOALLBenignImages
from DDSM_2_Folders import WOBenignImages
from DDSM_2_Folders import WOBenignWCImages
from DDSM_2_Folders import WOMalignantImages

from DDSM_2_Folders import NOALLBenignImages
from DDSM_2_Folders import NOBenignImages
from DDSM_2_Folders import NOBenignWCImages
from DDSM_2_Folders import NOMalignantImages

from DDSM_2_Folders import CLAHEALLBenignImages
from DDSM_2_Folders import CLAHEBenignImages
from DDSM_2_Folders import CLAHEBenignWCImages
from DDSM_2_Folders import CLAHEMalignantImages

from DDSM_2_Folders import HEALLBenignImages
from DDSM_2_Folders import HEBenignImages
from DDSM_2_Folders import HEBenignWCImages
from DDSM_2_Folders import HEMalignantImages

from DDSM_2_Folders import UMALLBenignImages
from DDSM_2_Folders import UMBenignImages
from DDSM_2_Folders import UMBenignWCImages
from DDSM_2_Folders import UMMalignantImages

from DDSM_2_Folders import CSALLBenignImages
from DDSM_2_Folders import CSBenignImages
from DDSM_2_Folders import CSBenignWCImages
from DDSM_2_Folders import CSMalignantImages


from DDSM_6_Data_Augmentation import DataAugmentation

Normal = 2
Tumor = 1

NNormal = 'BenignWC'
NTumor = 'Malignant'

IN = 0
IT = 1

#####

Images_BenignWC, Labels_BenignWC = DataAugmentation(WOBenignWCImages, NNormal, Normal, IN)

Images_Malignant, Labels_Malignant = DataAugmentation(WOMalignantImages, NTumor, Tumor, IT)

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

#####

print(len(Images_BenignWC))
print(len(Images_Malignant))