import os
import cv2
import random
import albumentations as A

from skimage import img_as_ubyte
from skimage import io
from skimage import filters

# Data Augmentation

def ShiftRotation(Image_Cropped):

    """
	  Shift rotation using albumentation.

    Parameters:
    argument1 (int): Image chosen.

    Returns:
	  int:Returning image with shift rotation applied
    
   	"""

    transform = A.Compose([
          A.ShiftScaleRotate(p = 1)
      ])
    transformed = transform(image = Image_Cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

def FlipHorizontal(Image_Cropped):

    """
	  Horizontal flip using albumentation.

    Parameters:
    argument1 (int): Image chosen.

    Returns:
	  int:Returning image with horizontal flip applied
    
   	"""
    transform = A.Compose([
        A.HorizontalFlip(p = 1)
      ])
    transformed = transform(image = Image_Cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

def FlipVertical(Image_Cropped):

    """
	  Vertical flip using albumentation.

    Parameters:
    argument1 (int): Image chosen.

    Returns:
	  int:Returning image with vertical flip applied
    
   	"""

    transform = A.Compose([
          A.VerticalFlip(p = 1)
        ])
    transformed = transform(image = Image_Cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

def Rotation(Rotation, Image_Cropped):

    """
	  Rotation using albumentation.

    Parameters:
    argument1 (float): Degrees of rotation.
    argument2 (int): Image chosen.

    Returns:
	  int:Returning image with rotation applied
    
   	"""


    transform = A.Compose([
        A.Rotate(Rotation, p = 1)
      ])
    transformed = transform(image = Image_Cropped)
    Imagen_transformada = transformed["image"]

    return Imagen_transformada

def DataAugmentation(Path_Folder, Severity, Sampling, Label):

    """
	  Applying data augmentation different transformations.

    Parameters:
    argument1 (folder): Folder chosen.
    argument2 (str): Severity of each image.
    argument3 (int): Amount of transformation applied for each image, using only rotation.
    argument4 (str): Label for each image.

    Returns:
	  list:Returning images like 'X' value
    list:Returning labels like 'Y' value
    
   	"""

    #All_data_dir = 'Mammography_data_' + str(Technique)

    #New_Folder_Data = os.path.join(New_Path_Folder, All_data_dir)

    #os.mkdir(New_Folder_Data)
    #print("Directory '% s' created" % All_data_dir)

    Images = [] 
    Labels = [] 

    png = ".png"

    #width = 224
    #height = 224

    os.chdir(Path_Folder)
    count = 1

    #dim = (width, height)

    images = len(os.listdir(Path_Folder))

    for File in os.listdir():

      filename, extension  = os.path.splitext(File)

      if File.endswith(png): # Read png files

        print(f"Working with {count} of {images} images of {Severity}")
        count += 1

        Path_File = os.path.join(Path_Folder, File)
        Resize_Imagen = cv2.imread(Path_File)
        #Resize_Imagen = cv2.cvtColor(Resize_Imagen, cv2.COLOR_BGR2GRAY)
        #Resize_Imagen = cv2.resize(Resize_Imagen, dim, interpolation = cv2.INTER_CUBIC)

        # 1) Raw

        Images.append(Resize_Imagen)
        Labels.append(Label)

        # 1.a) Rotation

        for i in range(Sampling):

          Imagen_transformed = Rotation(random.randint(-180, 180), Resize_Imagen)

          Images.append(Imagen_transformed)
          Labels.append(Label)

          #FilenamesREFNUM = filename + '_' + str(i) + '_Rotation' + '_Augmentation'
          #dst = FilenamesREFNUM + png

          #dstPath = os.path.join(Path_Folder, dst)
          #io.imsave(dstPath, Imagen_transformed)

        # 1.b) Flip Horizontal

        Imagen_transformed = FlipVertical(Resize_Imagen)

        Images.append(Imagen_transformed)
        Labels.append(Label)

        #FilenamesREFNUM = filename + '_' + str(i) + '_FlipVertical' + '_Augmentation'
        #dst = FilenamesREFNUM + png

        #dstPath = os.path.join(Path_Folder, dst)
        #io.imsave(dstPath, Imagen_transformed)

        # 1.c) Flip Vertical 

        Imagen_transformed = FlipHorizontal(Resize_Imagen)

        Images.append(Imagen_transformed)
        Labels.append(Label)
      
        print(len(Labels))

        #FilenamesREFNUM = filename + '_' + str(i) + '_FlipHorizontal' + '_Augmentation'
        #dst = FilenamesREFNUM + png

        #dstPath = os.path.join(Path_Folder, dst)
        #io.imsave(dstPath, Imagen_transformed)

    return Images, Labels