
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import GeneralDataCSV

from DDSM_2_Folders import HEAbnormalImages
from DDSM_2_Folders import HEAbnormalMassImages

from DDSM_2_Folders import UMAbnormalImages
from DDSM_2_Folders import UMAbnormalMassImages

from DDSM_2_Folders import CSAbnormalImages
from DDSM_2_Folders import CSAbnormalMassImages

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

HEImages_Calcification, HELabels_Calcification = DataAugmentation(HEAbnormalImages, NCalcification, Calcification, IN)

HEImages_Mass, HELabels_Mass = DataAugmentation(HEAbnormalMassImages, NMass, Mass, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

UMImages_Calcification, UMLabels_Calcification = DataAugmentation(UMAbnormalImages, NCalcification, Calcification, IN)

UMImages_Mass, UMLabels_Mass = DataAugmentation(UMAbnormalMassImages, NMass, Mass, IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

CSImages_Calcification, CSLabels_Calcification = DataAugmentation(CSAbnormalImages, NCalcification, Calcification, IN)

CSImages_Mass, CSLabels_Mass = DataAugmentation(CSAbnormalMassImages, NMass, Mass , IT)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

BiclassPrinting(HEImages_Calcification, HEImages_Mass, HETechnique)
BiclassPrinting(UMImages_Calcification, UMImages_Mass, UMTechnique)
BiclassPrinting(CSImages_Calcification, CSImages_Mass, CSTechnique)

