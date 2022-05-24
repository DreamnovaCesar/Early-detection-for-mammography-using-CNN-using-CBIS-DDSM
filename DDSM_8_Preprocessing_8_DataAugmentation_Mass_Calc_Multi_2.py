
########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

from DDSM_2_Folders import GeneralDataCSV

from DDSM_2_Folders import HENormalImages
from DDSM_2_Folders import HEAbnormalImages
from DDSM_2_Folders import HEAbnormalMassImages

from DDSM_2_Folders import UMNormalImages
from DDSM_2_Folders import UMAbnormalImages
from DDSM_2_Folders import UMAbnormalMassImages

from DDSM_2_Folders import CSNormalImages
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

HEImages_Normal, HELabels_Normal = DataAugmentation(HENormalImages, NNormal, Normal, IN)

HEImages_Mass, HELabels_Mass = DataAugmentation(HEAbnormalMassImages, NMass, Calcification, IM)

HEImages_Calcification, HELabels_Calcification = DataAugmentation(HEAbnormalImages, NCalcification, Calcification, IC)


########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

UMImages_Normal, UMLabels_Normal = DataAugmentation(UMNormalImages, NNormal, Normal, IN)

UMImages_Mass, UMLabels_Mass = DataAugmentation(UMAbnormalMassImages, NMass, Mass, IM)

UMImages_Calcification, UMLabels_Calcification = DataAugmentation(UMAbnormalImages, NCalcification, Calcification, IC)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

CSImages_Normal, CSLabels_Normal = DataAugmentation(CSNormalImages, NNormal, Normal, IN)

CSImages_Mass, CSLabels_Mass = DataAugmentation(CSAbnormalMassImages, NMass, Mass, IM)

CSImages_Calcification, CSLabels_Calcification = DataAugmentation(CSAbnormalImages, NCalcification, Calcification, IC)

########## ########## ########## ########## ########## ########## ########## ########## ########## ##########

TriclassPrinting(HEImages_Normal, HEImages_Mass, HEImages_Calcification, HETechnique)
TriclassPrinting(UMImages_Normal, UMImages_Mass, UMImages_Calcification, UMTechnique)
TriclassPrinting(CSImages_Normal, CSImages_Mass, CSImages_Calcification, CSTechnique)
