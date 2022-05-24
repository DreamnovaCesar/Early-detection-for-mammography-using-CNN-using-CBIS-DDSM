from DDSM_6_Data_Augmentation import DataAugmentation

from DDSM_2_Folders import GeneralDataCSV

from DDSM_2_Folders import WOAbnormalImages
from DDSM_2_Folders import WOAbnormalMassImages

from DDSM_2_Folders import NONormalImages
from DDSM_2_Folders import NOBenignImages
from DDSM_2_Folders import NOMalignantImages
from DDSM_2_Folders import NOBenignMassImages
from DDSM_2_Folders import NOMalignantMassImages


from DDSM_2_Folders import CLAHEAbnormalImages
from DDSM_2_Folders import CLAHEAbnormalMassImages

Normal = 15
BenignCalc = 2
MalignantCalc = 2
BenignMass = 2
MalignantMass = 2

NNormal = 'Normal'
NBenignCalc = 'Calcification'
NMalignantCalc = 'Calcification'
NBenignMass = 'Mass'
NMalignantMass = 'Mass'

IN = 0
IBC = 1
IMC = 2
IBM = 3
IMM = 4

#####

Images_Normal, Labels_Normal = DataAugmentation(NONormalImages, NNormal, Normal, IN)

Images_BCalcification, Labels_BCalcification = DataAugmentation(NOBenignImages, NBenignCalc, BenignCalc, IBC)

Images_MCalcification, Labels_MCalcification = DataAugmentation(NOMalignantImages, NMalignantCalc, MalignantCalc, IMC)

Images_BMass, Labels_BMass = DataAugmentation(NOBenignMassImages, NBenignMass, BenignMass, IBM)

Images_MMass, Labels_MMass = DataAugmentation(NOMalignantMassImages, NMalignantMass, MalignantMass, IMM)

"""
#####

NOImages_Calcification, NOLabels_Calcification = DataAugmentation(NOAbnormalImages, NNormal, Normal, IN)

NOImages_Mass, NOLabels_Mass = DataAugmentation(NOAbnormalMassImages, NTumor, Tumor, IT)

#####

CLAHEImages_Calcification, CLAHELabels_Calcification = DataAugmentation(CLAHEAbnormalImages, NNormal, Normal, IN)

CLAHEImages_Mass, CLAHELabels_Mass = DataAugmentation(CLAHEAbnormalMassImages, NTumor, Tumor, IT)

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

print(len(Images_Normal))
print(len(Images_BCalcification))
print(len(Images_MCalcification))
print(len(Images_BMass))
print(len(Images_MMass))


