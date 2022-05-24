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

Benign = 2
BenignWc = 3
Malignant = 2

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
Images_Benign, Labels_Benign = DataAugmentation(WOBenignImages299, NBenign, Benign, IN)

Images_BenignWC, Labels_BenignWC = DataAugmentation(WOBenignWCImages299, NBenignWc, BenignWc, IB)

Images_Malignant, Labels_Malignant = DataAugmentation(WOMalignantImages299, NMalignant, Malignant, IM)
"""
#####

NOImages_Benign, NOLabels_Benign = DataAugmentation(NOBenignImages299, NBenign, Benign, IN)

NOImages_BenignWC, NOLabels_BenignWC = DataAugmentation(NOBenignWCImages299, NBenignWc, BenignWc, IB)

NOImages_Malignant, NOLabels_Malignant = DataAugmentation(NOMalignantImages299, NMalignant, Malignant, IM)
"""
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


MFCLAHEImages_Benign, MFCLAHELabels_Benign = DataAugmentation(MFCLAHEBenignImages, NBenign, Benign, IN)

MFCLAHEImages_BenignWC, MFCLAHELabels_BenignWC = DataAugmentation(MFCLAHEBenignWCImages, NBenignWc, BenignWc, IB)

MFCLAHEImages_Malignant, MFCLAHELabels_Malignant = DataAugmentation(MFCLAHEMalignantImages, NMalignant, Malignant, IM)
"""
#####

print(len(NOImages_Benign))
print(len(NOImages_BenignWC))
print(len(NOImages_Malignant))
