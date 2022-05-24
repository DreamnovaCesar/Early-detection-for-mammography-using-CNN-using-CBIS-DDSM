
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import GeneralDataCSV

from DDSM_2_Folders import WOAbnormalImages
from DDSM_2_Folders import WOAbnormalMassImages

from DDSM_2_Folders import NOAbnormalImages
from DDSM_2_Folders import NOAbnormalMassImages

from DDSM_2_Folders import CLAHEAbnormalImages
from DDSM_2_Folders import CLAHEAbnormalMassImages

from DDSM_2_Folders import DataAugmentationFolder

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_8_Preprocessing_9_Parameters import RAWTechnique
from DDSM_8_Preprocessing_9_Parameters import NOTechnique
from DDSM_8_Preprocessing_9_Parameters import CLAHETechnique
from DDSM_8_Preprocessing_9_Parameters import HETechnique
from DDSM_8_Preprocessing_9_Parameters import UMTechnique
from DDSM_8_Preprocessing_9_Parameters import CSTechnique

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_4_DDSM_Functions import BiclassPrinting
from DDSM_6_Data_Augmentation import DataAugmentation

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Calcification = 3
Mass = 2

NCalcification = 'Calcification'
NMass = 'Mass'

IN = 0 
IT = 1

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

Images_Calcification, Labels_Calcification = DataAugmentation(WOAbnormalImages, NCalcification, Calcification, IN)

Images_Mass, Labels_Mass = DataAugmentation(WOAbnormalMassImages, NMass, Mass, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

NOImages_Calcification, NOLabels_Calcification = DataAugmentation(NOAbnormalImages, NCalcification, Calcification, IN)

NOImages_Mass, NOLabels_Mass = DataAugmentation(NOAbnormalMassImages, NMass, Mass, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

CLAHEImages_Calcification, CLAHELabels_Calcification = DataAugmentation(CLAHEAbnormalImages,  NCalcification, Calcification, IN)

CLAHEImages_Mass, CLAHELabels_Mass = DataAugmentation(CLAHEAbnormalMassImages, NMass, Mass, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

BiclassPrinting(Images_Calcification, Images_Mass, RAWTechnique)
BiclassPrinting(NOImages_Calcification, NOImages_Mass, NOTechnique)
BiclassPrinting(CLAHEImages_Calcification, CLAHEImages_Mass, CLAHETechnique)
