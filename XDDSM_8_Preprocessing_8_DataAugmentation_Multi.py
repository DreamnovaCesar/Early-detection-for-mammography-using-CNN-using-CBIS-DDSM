import cv2
import numpy as np

from sklearn.model_selection import train_test_split

from DDSM_6_Data_Augmentation import DataAugmentation

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

from DDSM_2_Folders import MFCLAHEALLBenignImages
from DDSM_2_Folders import MFCLAHEBenignImages
from DDSM_2_Folders import MFCLAHEBenignWCImages
from DDSM_2_Folders import MFCLAHEMalignantImages

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


Benign = 6
BenignWc = 8
Malignant = 6

#Benign = 14
#BenignWc = 18
#Malignant = 14

NBenign = 'Normal'
NBenignWc = 'Benign'
NMalignant = 'Malignant'

IN = 0
IB = 1
IM = 2
"""
#####
Images_Benign, Labels_Benign = DataAugmentation(WOBenignImages, NBenign, Benign, IN)

Images_BenignWC, Labels_BenignWC = DataAugmentation(WOBenignWCImages, NBenignWc, BenignWc, IB)

Images_Malignant, Labels_Malignant = DataAugmentation(WOMalignantImages, NMalignant, Malignant, IM)

#####

NOImages_Benign, NOLabels_Benign = DataAugmentation(NOBenignImages, NBenign, Benign, IN)

NOImages_BenignWC, NOLabels_BenignWC = DataAugmentation(NOBenignWCImages, NBenignWc, BenignWc, IB)

NOImages_Malignant, NOLabels_Malignant = DataAugmentation(NOMalignantImages, NMalignant, Malignant, IM)

#####

CLAHEImages_Benign, CLAHELabels_Benign = DataAugmentation(MFCLAHEBenignImages, NBenign, Benign, IN)

CLAHEImages_BenignWC, CLAHELabels_BenignWC = DataAugmentation(MFCLAHEBenignWCImages, NBenignWc, BenignWc, IB)

CLAHEImages_Malignant, CLAHELabels_Malignant = DataAugmentation(MFCLAHEMalignantImages, NMalignant, Malignant, IM)

#####

HEImages_Benign, HELabels_Benign = DataAugmentation(HEBenignImages, NBenign, Benign, IN)

HEImages_BenignWC, HELabels_BenignWC = DataAugmentation(HEBenignWCImages, NBenignWc, BenignWc, IB)

HEImages_Malignant, HELabels_Malignant = DataAugmentation(HEMalignantImages, NMalignant, Malignant, IM)

#####

UMImages_Benign, UMLabels_Benign = DataAugmentation(UMBenignImages, NBenign, Benign, IN)

UMImages_BenignWC, UMLabels_BenignWC = DataAugmentation(UMBenignWCImages, NBenignWc, BenignWc, IB)

UMImages_Malignant, UMLabels_Malignant = DataAugmentation(UMMalignantImages, NMalignant, Malignant, IM)

#####

CSImages_Benign, CSLabels_Benign = DataAugmentation(CSBenignImages, NBenign, Benign, IN)

CSImages_BenignWC, CSLabels_BenignWC = DataAugmentation(CSBenignWCImages, NBenignWc, BenignWc, IB)

CSImages_Malignant, CSLabels_Malignant = DataAugmentation(CSMalignantImages, NMalignant, Malignant, IM)

"""
MFCLAHEImages_Benign, MFCLAHELabels_Benign = DataAugmentation(MFCLAHEBenignImages, NBenign, Benign, IN)

MFCLAHEImages_BenignWC, MFCLAHELabels_BenignWC = DataAugmentation(MFCLAHEBenignWCImages, NBenignWc, BenignWc, IB)

MFCLAHEImages_Malignant, MFCLAHELabels_Malignant = DataAugmentation(MFCLAHEMalignantImages, NMalignant, Malignant, IM)
"""
#####

print(len(Images_Benign))
print(len(Images_BenignWC))
print(len(Images_Malignant))
"""