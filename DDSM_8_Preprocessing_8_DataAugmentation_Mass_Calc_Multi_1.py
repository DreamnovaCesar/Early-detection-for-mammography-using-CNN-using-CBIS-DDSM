
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import GeneralDataCSV

from DDSM_2_Folders import WONormalImages
from DDSM_2_Folders import WOAbnormalImages
from DDSM_2_Folders import WOAbnormalMassImages

from DDSM_2_Folders import NONormalImages
from DDSM_2_Folders import NOAbnormalImages
from DDSM_2_Folders import NOAbnormalMassImages

from DDSM_2_Folders import CLAHENormalImages
from DDSM_2_Folders import CLAHEAbnormalImages
from DDSM_2_Folders import CLAHEAbnormalMassImages

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_9_Parameters import RAWTechnique
from DDSM_8_Preprocessing_9_Parameters import NOTechnique
from DDSM_8_Preprocessing_9_Parameters import CLAHETechnique
from DDSM_8_Preprocessing_9_Parameters import HETechnique
from DDSM_8_Preprocessing_9_Parameters import UMTechnique
from DDSM_8_Preprocessing_9_Parameters import CSTechnique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_4_DDSM_Functions import TriclassPrinting
from DDSM_6_Data_Augmentation import DataAugmentation

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Normal = 30
Mass = 1
Calcification = 2

NNormal = 'Normal'
NMass = 'Mass'
NCalcification = 'Calcification'

IN = 0
IM = 1
IC = 2

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Images_Normal, Labels_Normal = DataAugmentation(WONormalImages, NNormal, Normal, IN)

Images_Mass, Labels_Mass = DataAugmentation(WOAbnormalMassImages, NMass, Mass, IM)

Images_Calcification, Labels_Calcification = DataAugmentation(WOAbnormalImages, NCalcification, Calcification, IC)


########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

NOImages_Normal, NOLabels_Normal = DataAugmentation(NONormalImages, NNormal, Normal, IN)

NOImages_Mass, NOLabels_Mass = DataAugmentation(NOAbnormalMassImages, NMass, Mass, IM)

NOImages_Calcification, NOLabels_Calcification = DataAugmentation(NOAbnormalImages, NCalcification, Calcification, IC)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

CLAHEImages_Normal, CLAHELabels_Normal = DataAugmentation(CLAHENormalImages, NNormal, Normal, IN)

CLAHEImages_Mass, CLAHELabels_Mass = DataAugmentation(CLAHEAbnormalMassImages, NMass, Mass, IM)

CLAHEImages_Calcification, CLAHELabels_Calcification = DataAugmentation(CLAHEAbnormalImages, NCalcification, Calcification, IC)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

TriclassPrinting(Images_Normal, Images_Mass, Images_Calcification, RAWTechnique)
TriclassPrinting(NOImages_Normal, NOImages_Mass, NOImages_Calcification, NOTechnique)
TriclassPrinting(CLAHEImages_Normal, CLAHEImages_Mass, CLAHEImages_Calcification, CLAHETechnique)

