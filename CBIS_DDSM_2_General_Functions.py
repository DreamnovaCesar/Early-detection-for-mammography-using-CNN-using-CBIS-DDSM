
import os
import cv2
import numpy as np
import pandas as pd
import pydicom
import tensorflow as tf

from sklearn.utils import resample

from glrlm import GLRLM

from skimage.feature import graycomatrix, graycoprops

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Detect fi GPU exist in your PC for CNN

def detect_GPU():

  GPU_name = tf.test.gpu_device_name()
  GPU_available = tf.test.is_gpu_available()

  print(GPU_available)

  if GPU_available == True:
      print("GPU device is available")

  if "GPU" not in GPU_name:
      print("GPU device not found")
  print('Found GPU at: {}'.format(GPU_name))

# Sort Files

def ShowSort(Folder_Path): 

	"""
	Read all images in a folder and sort them.

    Parameters:
    argument1 (Folder): Folder used.

    Returns:
	int:Returning value
    int:Returning list[str]

   	"""

	NumberImages = len(os.listdir(Folder_Path))

	print("\n")
	print("********************************")
	print(f"Images: {NumberImages}")
	print("********************************")
	print("\n")

	files = os.listdir(Folder_Path)
	print(files)
	print("\n")

	print("********************************")
	sorted_files =  sorted(files)
	print(sorted_files)
	print("\n")
	print("********************************")

	return sorted_files, NumberImages

# Remove all files in folder

def Removeallfiles(Folder_Path):

	"""
	Remove all images inside the folder chosen

    Parameters:
    argument1 (Folder): Folder used.

    Returns:
	Void

   	"""

	for File in os.listdir(Folder_Path):
		filename, extension  = os.path.splitext(File)
		print(f"Removing {filename} ✅")
		os.remove(os.path.join(Folder_Path, File))

def TransformedLabelDDSM(Dataset_Cancer, Column):

  Dataset_Cancer.iloc[:, Column].values
  LE = LabelEncoder()
  Dataset_Cancer.iloc[:, Column] = LE.fit_transform(Dataset_Cancer.iloc[:, Column])
  Dataset_Calc_Pathology = Dataset_Cancer.iloc[:, Column].values
  Dataset = Dataset_Cancer.iloc[:, Column]

  return Dataset_Calc_Pathology, Dataset

def SplitDataDDSMMod(Folder_Path, Folder_Path_Benign_Total, Folder_Path_Benign, Folder_Path_BenignWC, Folder_Path_Malignant, Dataset):

  Removeallfiles(Folder_Path_Benign_Total)
  Removeallfiles(Folder_Path_Benign)
  Removeallfiles(Folder_Path_BenignWC)
  Removeallfiles(Folder_Path_Malignant)

  Images = []
  Label = []

  Filename_Benign_All = []
  Filename_Malignant_All = []
  Filename_All = []

  Filename_Benign = []
  Filename_Benign_WC = []
  Filename_Malignant = []

  Benign = 0
  Benign_with_callback = 1
  Malignant = 2

  Index = 0

  png = ".png"    # png.
    
  os.chdir(Folder_Path)

  sorted_files, images = ShowSort(Folder_Path)
  Index = 0
  count = 1

  for File in sorted_files:
    
    filename, extension  = os.path.splitext(File)

    if Dataset[Index] == Benign:

        try:

            print(f"Working with {count} of {images} Benign images, {filename}")
            count += 1

            Path_File = os.path.join(Folder_Path, File)
            Imagen_Benign = cv2.imread(Path_File)

            FileName_Benign = filename + '_benign'
            dst_name = FileName_Benign + png

            dstPath_name = os.path.join(Folder_Path_Benign_Total, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            dstPath_name = os.path.join(Folder_Path_Benign, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            Images.append(Imagen_Benign)
            Label.append(Benign)

            Filename_Benign_All.append(FileName_Benign)
            Filename_Benign.append(FileName_Benign)
            Filename_All.append(FileName_Benign)

        except OSError:
            print('Cannot convert %s' % File)

    elif Dataset[Index] == Benign_with_callback:
  

            try:

                print(f"Working with {count} of {images} Benign_without_callback images, {filename}")
                count += 1

                Path_File = os.path.join(Folder_Path, File)
                Imagen_Benign_without_callback = cv2.imread(Path_File)

                FileName_Benign_WC = filename + '_benign_without_callback'
                dst_name = FileName_Benign_WC + png

                dstPath_name = os.path.join(Folder_Path_Benign_Total, dst_name)
                cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                dstPath_name = os.path.join(Folder_Path_BenignWC, dst_name)
                cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                Images.append(Imagen_Benign_without_callback)
                Label.append(Benign_with_callback)

                Filename_Benign_All.append(FileName_Benign_WC)
                Filename_Benign_WC.append(FileName_Benign_WC)
                Filename_All.append(FileName_Benign_WC)

            except OSError:
                print('Cannot convert %s' % File)
      
    elif Dataset[Index] == Malignant:


            try:

                print(f"Working with {count} of {images} Malignant images, {filename}")
                count += 1

                Path_File = os.path.join(Folder_Path, File)
                Imagen_Malignant = cv2.imread(Path_File)

                FileName_Malignant = filename + '_malignant'
                dst_name = FileName_Malignant + png

                dstPath_name = os.path.join(Folder_Path_Malignant, dst_name)
                cv2.imwrite(dstPath_name,  Imagen_Malignant)

                Images.append(Imagen_Malignant)
                Label.append(Malignant)

                Filename_Malignant_All.append(FileName_Malignant)
                Filename_Malignant.append(FileName_Malignant)
                Filename_All.append(FileName_Malignant)

            except OSError:
                print('Cannot convert %s' % File)
                
    Index += 1

  Dataset = pd.DataFrame({'Filename_All':Filename_All,'Labels':Label}) 

  return Images, Dataset

def SplitDataDDSM(Folder_Path_Test, Folder_Path_Training, Folder_Path_Benign_Total, Folder_Path_Benign, Folder_Path_BenignWC, Folder_Path_Malignant, Folder_Path_Abnormal, Dataset_Test, Dataset_Training):

  Removeallfiles(Folder_Path_Benign_Total)
  Removeallfiles(Folder_Path_Benign)
  Removeallfiles(Folder_Path_BenignWC)
  Removeallfiles(Folder_Path_Malignant)
  Removeallfiles(Folder_Path_Abnormal)

  Images = []
  Label = []

  Filename_Benign_All = []
  Filename_Malignant_All = []
  Filename_All = []

  Filename_Benign = []
  Filename_Benign_WC = []
  Filename_Malignant = []
  Filename_Abnormal = []

  Benign = 0
  Benign_with_callback = 1
  Malignant = 2

  png = ".png"    # png.

  os.chdir(Folder_Path_Test)

  sorted_files, images = ShowSort(Folder_Path_Test)
  Index = 0
  count = 1
  
  for File in sorted_files:
    
    filename, extension  = os.path.splitext(File)
    
    if Dataset_Test[Index] == Benign:

        try:

            print(f"Working with {count} of {images} Benign images, {filename}")
            count += 1

            Path_File = os.path.join(Folder_Path_Test, File)
            Imagen_Benign = cv2.imread(Path_File)

            FileName_Benign = filename + '_benign'
            dst_name = FileName_Benign + png

            dstPath_name = os.path.join(Folder_Path_Abnormal, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            dstPath_name = os.path.join(Folder_Path_Benign_Total, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            dstPath_name = os.path.join(Folder_Path_Benign, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            #Images.append(Imagen_Benign)
            Label.append(Benign)

            Filename_Benign_All.append(FileName_Benign)
            Filename_Benign.append(FileName_Benign)
            Filename_All.append(FileName_Benign)

        except OSError:
            print('Cannot convert %s' % File)

    elif Dataset_Test[Index] == Benign_with_callback:
  

            try:

                print(f"Working with {count} of {images} Benign_without_callback images, {filename}")
                count += 1

                Path_File = os.path.join(Folder_Path_Test, File)
                Imagen_Benign_without_callback = cv2.imread(Path_File)

                FileName_Benign_WC = filename + '_benign_without_callback'
                dst_name = FileName_Benign_WC + png

                dstPath_name = os.path.join(Folder_Path_Benign_Total, dst_name)
                cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                #dstPath_name = os.path.join(Folder_Path_Abnormal, dst_name)
                #cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                dstPath_name = os.path.join(Folder_Path_BenignWC, dst_name)
                cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                #Images.append(Imagen_Benign_without_callback)
                Label.append(Benign_with_callback)

                Filename_Benign_All.append(FileName_Benign_WC)
                Filename_Benign_WC.append(FileName_Benign_WC)
                Filename_All.append(FileName_Benign_WC)

            except OSError:
                print('Cannot convert %s' % File)
      
    elif Dataset_Test[Index] == Malignant:


            try:

                print(f"Working with {count} of {images} Malignant images, {filename}")
                count += 1

                Path_File = os.path.join(Folder_Path_Test, File)
                Imagen_Malignant = cv2.imread(Path_File)

                FileName_Malignant = filename + '_malignant'
                dst_name = FileName_Malignant + png

                dstPath_name = os.path.join(Folder_Path_Abnormal, dst_name)
                cv2.imwrite(dstPath_name,  Imagen_Malignant)

                dstPath_name = os.path.join(Folder_Path_Malignant, dst_name)
                cv2.imwrite(dstPath_name,  Imagen_Malignant)

                #Images.append(Imagen_Malignant)
                Label.append(Malignant)

                Filename_Malignant_All.append(FileName_Malignant)
                Filename_Malignant.append(FileName_Malignant)
                Filename_All.append(FileName_Malignant)

            except OSError:
                print('Cannot convert %s' % File)

    Index += 1
    print(Index)
    
  os.chdir(Folder_Path_Training)

  sorted_files, images = ShowSort(Folder_Path_Training)
  Index = 0
  count = 1

  for File in sorted_files:
    
    filename, extension  = os.path.splitext(File)

    if Dataset_Training[Index] == Benign:

        try:

            print(f"Working with {count} of {images} Benign images, {filename}")
            count += 1

            Path_File = os.path.join(Folder_Path_Training, File)
            Imagen_Benign = cv2.imread(Path_File)

            FileName_Benign = filename + '_benign'
            dst_name = FileName_Benign + png

            dstPath_name = os.path.join(Folder_Path_Benign_Total, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            dstPath_name = os.path.join(Folder_Path_Abnormal, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            dstPath_name = os.path.join(Folder_Path_Benign, dst_name)
            cv2.imwrite(dstPath_name, Imagen_Benign)

            #Images.append(Imagen_Benign)
            Label.append(Benign)

            Filename_Benign_All.append(FileName_Benign)
            Filename_Benign.append(FileName_Benign)
            Filename_All.append(FileName_Benign)

        except OSError:
            print('Cannot convert %s' % File)

    elif Dataset_Training[Index] == Benign_with_callback:
  

            try:

                print(f"Working with {count} of {images} Benign_without_callback images, {filename}")
                count += 1

                Path_File = os.path.join(Folder_Path_Training, File)
                Imagen_Benign_without_callback = cv2.imread(Path_File)

                FileName_Benign_WC = filename + '_benign_without_callback'
                dst_name = FileName_Benign_WC + png

                dstPath_name = os.path.join(Folder_Path_Benign_Total, dst_name)
                cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                dstPath_name = os.path.join(Folder_Path_BenignWC, dst_name)
                cv2.imwrite(dstPath_name, Imagen_Benign_without_callback)

                #Images.append(Imagen_Benign_without_callback)
                Label.append(Benign_with_callback)

                Filename_Benign_All.append(FileName_Benign_WC)
                Filename_Benign_WC.append(FileName_Benign_WC)
                Filename_All.append(FileName_Benign_WC)

            except OSError:
                print('Cannot convert %s' % File)
      
    elif Dataset_Training[Index] == Malignant:


            try:

                print(f"Working with {count} of {images} Malignant images, {filename}")
                count += 1

                Path_File = os.path.join(Folder_Path_Training, File)
                Imagen_Malignant = cv2.imread(Path_File)

                FileName_Malignant = filename + '_malignant'
                dst_name = FileName_Malignant + png

                dstPath_name = os.path.join(Folder_Path_Abnormal, dst_name)
                cv2.imwrite(dstPath_name,  Imagen_Malignant)

                dstPath_name = os.path.join(Folder_Path_Malignant, dst_name)
                cv2.imwrite(dstPath_name,  Imagen_Malignant)

                #Images.append(Imagen_Malignant)
                Label.append(Malignant)

                Filename_Malignant_All.append(FileName_Malignant)
                Filename_Malignant.append(FileName_Malignant)
                Filename_All.append(FileName_Malignant)

            except OSError:
                print('Cannot convert %s' % File)
                
    Index += 1

  Dataset = pd.DataFrame({'Filename_All':Filename_All,'Labels':Label}) 

  return Images, Dataset

def DatasetSplit(Dataset, X, Y):

  X = Dataset.drop('Labels', axis = 1)
  Y = Dataset['Labels']

  Majority, Minority = Dataset['Labels'].value_counts()

  return X, Y, Majority, Minority

def ImbalanceDataMajority(Dataset, X, Y):

  X = Dataset.drop('Labels', axis = 1)
  Y = Dataset['Labels']

  Majority, Minority = Y.value_counts()

  df_majority = Dataset[Y == 0]
  df_minority = Dataset[Y == 1]

  df_majority_downsampled = resample( df_majority, 
                                      replace = False,     # sample with replacement
                                      n_samples = Minority, # to match majority class
                                      random_state = 123) # reproducible results
  
  df_downsampled = pd.concat([df_minority, df_majority_downsampled])
  print(df_downsampled['Labels'].value_counts())

  X = df_downsampled.drop('Labels', axis = 1)
  Y = df_downsampled['Labels']

  return X, Y, Majority, Minority

def ImbalanceDataMinority(Dataset, X, Y):

  X = Dataset.drop('Labels', axis = 1)
  Y = Dataset['Labels']

  Majority, Minority = Y.value_counts()

  df_majority = Dataset[Y == 0]
  df_minority = Dataset[Y == 1]

  df_minority_upsampled = resample( df_minority, 
                                    replace = True,     # sample with replacement
                                    n_samples = Majority, # to match majority class
                                    random_state = 123) # reproducible results

  df_upsampled = pd.concat([df_majority, df_minority_upsampled])
  print(df_upsampled['Labels'].value_counts())

  X = df_upsampled.drop('Labels', axis = 1)
  Y = df_upsampled['Labels']

  return X, Y, Majority, Minority

#Tamura

def coarseness(image, kmax):

	image = np.array(image)
	w = image.shape[0]
	h = image.shape[1]
	kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
	kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))
	average_gray = np.zeros([kmax,w,h])
	horizon = np.zeros([kmax,w,h])
	vertical = np.zeros([kmax,w,h])
	Sbest = np.zeros([w,h])

	for k in range(kmax):
		window = np.power(2, k)
		for wi in range(w)[window:(w-window)]:
			for hi in range(h)[window:(h-window)]:
				average_gray[k][wi][hi] = np.sum(image[wi-window:wi+window, hi-window:hi+window])
		for wi in range(w)[window:(w-window-1)]:
			for hi in range(h)[window:(h-window-1)]:
				horizon[k][wi][hi] = average_gray[k][wi+window][hi] - average_gray[k][wi-window][hi]
				vertical[k][wi][hi] = average_gray[k][wi][hi+window] - average_gray[k][wi][hi-window]
		horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
		vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

	for wi in range(w):
		for hi in range(h):
			h_max = np.max(horizon[:,wi,hi])
			h_max_index = np.argmax(horizon[:,wi,hi])
			v_max = np.max(vertical[:,wi,hi])
			v_max_index = np.argmax(vertical[:,wi,hi])
			index = h_max_index if (h_max > v_max) else v_max_index
			Sbest[wi][hi] = np.power(2,index)

	fcrs = np.mean(Sbest)
	return fcrs

def contrast(image):
	image = np.array(image)
	image = np.reshape(image, (1, image.shape[0]*image.shape[1]))
	m4 = np.mean(np.power(image - np.mean(image),4))
	v = np.var(image)
	std = np.power(v, 0.5)
	alfa4 = m4 / np.power(v,2)
	fcon = std / np.power(alfa4, 0.25)
	return fcon

def directionality(image):
	image = np.array(image, dtype = 'int64')
	h = image.shape[0]
	w = image.shape[1]
	convH = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	convV = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	deltaH = np.zeros([h,w])
	deltaV = np.zeros([h,w])
	theta = np.zeros([h,w])

	# calc for deltaH
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaH[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convH))
	for wi in range(w)[1:w-1]:
		deltaH[0][wi] = image[0][wi+1] - image[0][wi]
		deltaH[h-1][wi] = image[h-1][wi+1] - image[h-1][wi]
	for hi in range(h):
		deltaH[hi][0] = image[hi][1] - image[hi][0]
		deltaH[hi][w-1] = image[hi][w-1] - image[hi][w-2]

	# calc for deltaV
	for hi in range(h)[1:h-1]:
		for wi in range(w)[1:w-1]:
			deltaV[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convV))
	for wi in range(w):
		deltaV[0][wi] = image[1][wi] - image[0][wi]
		deltaV[h-1][wi] = image[h-1][wi] - image[h-2][wi]
	for hi in range(h)[1:h-1]:
		deltaV[hi][0] = image[hi+1][0] - image[hi][0]
		deltaV[hi][w-1] = image[hi+1][w-1] - image[hi][w-1]

	deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
	deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

	# calc the theta
	for hi in range(h):
		for wi in range(w):
			if (deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0):
				theta[hi][wi] = 0;
			elif(deltaH[hi][wi] == 0):
				theta[hi][wi] = np.pi
			else:
				theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
	theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

	n = 16
	t = 12
	cnt = 0
	hd = np.zeros(n)
	dlen = deltaG_vec.shape[0]
	for ni in range(n):
		for k in range(dlen):
			if((deltaG_vec[k] >= t) and (theta_vec[k] >= (2*ni-1) * np.pi / (2 * n)) and (theta_vec[k] < (2*ni+1) * np.pi / (2 * n))):
				hd[ni] += 1
	hd = hd / np.mean(hd)
	hd_max_index = np.argmax(hd)
	fdir = 0
	for ni in range(n):
		fdir += np.power((ni - hd_max_index), 2) * hd[ni]
	return fdir

def linelikeness(image, sita, dist):
	pass

def regularity(image, filter):
	pass

def roughness(fcrs, fcon):
	return fcrs + fcon

# First Order features

def fos(f, mask):
    '''
    Parameters
    ----------
    f : numpy ndarray
        Image of dimensions N1 x N2.
    mask : numpy ndarray
        Mask image N1 x N2 with 1 if pixels belongs to ROI, 0 else. Give None
        if you want to consider ROI the whole image.
    Returns
    -------
    features : numpy ndarray
        1)Mean, 2)Variance, 3)Median (50-Percentile), 4)Mode, 
        5)Skewness, 6)Kurtosis, 7)Energy, 8)Entropy, 
        9)Minimal Gray Level, 10)Maximal Gray Level, 
        11)Coefficient of Variation, 12,13,14,15)10,25,75,90-
        Percentile, 16)Histogram width
    labels : list
        Labels of features.
    '''
    if mask is None:
        mask = np.ones(f.shape)
    
    # 1) Labels
    labels = ["FOS_Mean","FOS_Variance","FOS_Median","FOS_Mode","FOS_Skewness",
              "FOS_Kurtosis","FOS_Energy","FOS_Entropy","FOS_MinimalGrayLevel",
              "FOS_MaximalGrayLevel","FOS_CoefficientOfVariation",
              "FOS_10Percentile","FOS_25Percentile","FOS_75Percentile",
              "FOS_90Percentile","FOS_HistogramWidth"]
    
    # 2) Parameters
    f  = f.astype(np.uint8)
    mask = mask.astype(np.uint8)
    level_min = 0
    level_max = 255
    Ng = (level_max - level_min) + 1
    bins = Ng
    
    # 3) Calculate Histogram H inside ROI
    f_ravel = f.ravel() 
    mask_ravel = mask.ravel() 
    roi = f_ravel[mask_ravel.astype(bool)] 
    H = np.histogram(roi, bins = bins, range = [level_min, level_max], density = True)[0]
    
    # 4) Calculate Features
    features = np.zeros(16, np.double)  
    i = np.arange(0, bins)
    features[0] = np.dot(i, H)
    features[1] = sum(np.multiply((( i- features[0]) ** 2), H))
    features[2] = np.percentile(roi, 50) 
    features[3] = np.argmax(H)
    features[4] = sum(np.multiply(((i-features[0]) ** 3), H)) / (np.sqrt(features[1]) ** 3)
    features[5] = sum(np.multiply(((i-features[0]) ** 4), H)) / (np.sqrt(features[1]) ** 4)
    features[6] = sum(np.multiply(H, H))
    features[7] = -sum(np.multiply(H, np.log(H + 1e-16)))
    features[8] = min(roi)
    features[9] = max(roi)
    features[10] = np.sqrt(features[2]) / features[0]
    features[11] = np.percentile(roi, 10) 
    features[12] = np.percentile(roi, 25)  
    features[13] = np.percentile(roi, 75) 
    features[14] = np.percentile(roi, 90) 
    features[15] = features[14] - features[11]
    
    return features, labels

# First Order features Images

def TexturesFeatureFirstOrderImage(Images, Label):

  Fof = 'First Order Features'

  Mean = []
  Var = []
  Skew = []
  Kurtosis = []
  Energy = []
  Entropy = []
  Labels = []

  count = 1

  for File in range(len(Images)):

      try:

          print(f"Working with {count} of {len(Images)} images ✅")
          count += 1

          Features, Labels_ = fos(Images[File], None)

          Mean.append(Features[0])
          Var.append(Features[1])
          Skew.append(Features[4])
          Kurtosis.append(Features[5])
          Energy.append(Features[6])
          Entropy.append(Features[7])
          Labels.append(Label)

      except OSError:
          print('Cannot convert %s ❌' % File)

  Dataset = pd.DataFrame({'Mean':Mean, 'Var':Var, 'Kurtosis':Kurtosis, 'Energy':Energy, 'Skew':Skew, 'Entropy':Entropy, 'Labels':Labels})

  X = Dataset.iloc[:, [0, 1, 2, 3, 4, 5]].values
  Y = Dataset.iloc[:, -1].values

  return Dataset, X, Y, Fof

# GLRLM features Images

def TexturesFeatureGLRLMImage(Images, Label):

  Glrlm = 'Gray-Level Run Length Matrix'
  
  SRE = []  # Short Run Emphasis
  LRE  = [] # Long Run Emphasis
  GLU = []  # Grey Level Uniformity
  RLU = []  # Run Length Uniformity
  RPC = []  # Run Percentage
  Labels = []

  count = 1

  for File in range(len(Images)):

      try:
          print(f"Working with {count} of {len(Images)} images ✅")
          count += 1

          app = GLRLM()
          glrlm = app.get_features(Images[File], 8)

          SRE.append(glrlm.Features[0])
          LRE.append(glrlm.Features[1])
          GLU.append(glrlm.Features[2])
          RLU.append(glrlm.Features[3])
          RPC.append(glrlm.Features[4])
          Labels.append(Label)

      except OSError:
          print('Cannot convert %s ❌' % File)

  Dataset = pd.DataFrame({'SRE':SRE, 'LRE':LRE, 'GLU':GLU, 'RLU':RLU, 'RPC':RPC, 'Labels':Labels})

  X = Dataset.iloc[:, [0, 1, 2, 3, 4]].values
  Y = Dataset.iloc[:, -1].values

  return Dataset, X, Y, Glrlm

# GLCM features Images

def TexturesFeatureGLCMImage(Images, Label):

  Glcm = 'Gray-Level Co-Occurance Matrix'

  Dataset = pd.DataFrame()
  
  Dissimilarity = []
  Correlation = []
  Homogeneity = []
  Energy = []
  Contrast = []

  Dissimilarity2 = []
  Correlation2 = []
  Homogeneity2 = []
  Energy2 = []
  Contrast2 = []

  Dissimilarity3 = []
  Correlation3 = []
  Homogeneity3 = []
  Energy3 = []
  Contrast3 = []

  Dissimilarity4 = []
  Correlation4 = []
  Homogeneity4 = []
  Energy4 = []
  Contrast4 = []

  #Entropy = []
  #ASM = []
  Labels = []
  Labels2 = []
  Labels3 = []
  Labels4 = []

  count = 1

  for File in range(len(Images)):

      try:
          print(f"Working with {count} of {len(Images)} images ✅")
          count += 1

          Images[File] = cv2.cvtColor(Images[File], cv2.COLOR_BGR2GRAY)

          GLCM = graycomatrix(Images[File], [1], [0])
          Energy.append(graycoprops(GLCM, 'energy')[0, 0])
          Correlation.append(graycoprops(GLCM, 'correlation')[0, 0])
          Homogeneity.append(graycoprops(GLCM, 'homogeneity')[0, 0])
          Dissimilarity.append(graycoprops(GLCM, 'dissimilarity')[0, 0])
          Contrast.append(graycoprops(GLCM, 'contrast')[0, 0])
         
          GLCM2 = graycomatrix(Images[File], [5], [np.pi/4])
          Energy2.append(graycoprops(GLCM2, 'energy')[0, 0])
          Correlation2.append(graycoprops(GLCM2, 'correlation')[0, 0])
          Homogeneity2.append(graycoprops(GLCM2, 'homogeneity')[0, 0])
          Dissimilarity2.append(graycoprops(GLCM2, 'dissimilarity')[0, 0])
          Contrast2.append(graycoprops(GLCM2, 'contrast')[0, 0])

          GLCM3 = graycomatrix(Images[File], [7], [np.pi/2])
          Energy3.append(graycoprops(GLCM3, 'energy')[0, 0])
          Correlation3.append(graycoprops(GLCM3, 'correlation')[0, 0])
          Homogeneity3.append(graycoprops(GLCM3, 'homogeneity')[0, 0])
          Dissimilarity3.append(graycoprops(GLCM3, 'dissimilarity')[0, 0])
          Contrast3.append(graycoprops(GLCM3, 'contrast')[0, 0])

          GLCM4 = graycomatrix(Images[File], [7], [3 * np.pi/4])
          Energy4.append(graycoprops(GLCM4, 'energy')[0, 0])
          Correlation4.append(graycoprops(GLCM4, 'correlation')[0, 0])
          Homogeneity4.append(graycoprops(GLCM4, 'homogeneity')[0, 0])
          Dissimilarity4.append(graycoprops(GLCM4, 'dissimilarity')[0, 0])
          Contrast4.append(graycoprops(GLCM4, 'contrast')[0, 0])
         
          Labels.append(Label)
          # np.pi/4
          # np.pi/2
          # 3*np.pi/4

      except OSError:
          print('Cannot convert %s ❌' % File)
  
  """
  Dataset = pd.DataFrame({'Energy':Energy,  'Homogeneity':Homogeneity,  'Contrast':Contrast,  'Correlation':Correlation,
                          'Energy2':Energy, 'Homogeneity2':Homogeneity, 'Contrast2':Contrast, 'Correlation2':Correlation, 
                          'Energy3':Energy, 'Homogeneity3':Homogeneity, 'Contrast3':Contrast, 'Correlation3':Correlation, 
                          'Energy4':Energy, 'Homogeneity4':Homogeneity, 'Contrast4':Contrast, 'Correlation4':Correlation, 'Labels3':Labels})
  """
  Dataset = pd.DataFrame({'Energy':Energy,  'Homogeneity':Homogeneity,  'Contrast':Contrast,  'Correlation':Correlation,
                          'Energy2':Energy, 'Homogeneity2':Homogeneity, 'Contrast2':Contrast, 'Correlation2':Correlation, 'Labels3':Labels})


  #'Energy':Energy
  #'Homogeneity':Homogeneity
  #'Correlation':Correlation
  #'Contrast':Contrast
  #'Dissimilarity':Dissimilarity

  X = Dataset.iloc[:, [0, 1]].values
  Y = Dataset.iloc[:, -1].values

  return Dataset, X, Y, Glcm

# Biclass printing data augmentation

def BiclassPrinting(ImagesNormal, ImagesTumor, Technique):

    """
	  Printing amount of images with data augmentation 

    Parameters:
    argument1 (list): The number of Normal images.
    argument2 (list): The number of Tumor images.
    argument3 (str): Technique used

    Returns:
	  void
   	"""

    print("\n")
    print(Technique + ' images: ' + str(len(ImagesNormal)))
    print(Technique + ' images: ' + str(len(ImagesTumor)))

# Triclass printing data augmentation

def TriclassPrinting(ImagesNormal, ImagesBenign, ImagesMalignant, Technique):

    """
	  Printing amount of images with data augmentation 

    Parameters:
    argument1 (list): The number of Normal images.
    argument2 (list): The number of Benign images.
    argument3 (list): The number of Malignant images.
    argument4 (str): Technique used

    Returns:
	  void
    """

    print("\n")
    print(Technique + ' images: ' + str(len(ImagesNormal)))
    print(Technique + ' images: ' + str(len(ImagesBenign)))
    print(Technique + ' images: ' + str(len(ImagesMalignant)))

# Convertion DCM to PNG

def ConvertDCM(PATH_FILE, NAME, PathN, PathRe, PathReNo):

  """
	Printing amount of images with data augmentation 

  Parameters:
  argument1 (list): The number of Normal images.
  argument2 (list): The number of Benign images.
  argument3 (list): The number of Malignant images.
  argument4 (str): Technique used

  Returns:
	void
  """

  dcm = ".dcm"
  dcmfiles = []
  dcmfilessize = []
  Paths = []

  arr = os.listdir(PATH_FILE)

  Arr = arr * 2
  Arr = sorted(Arr)

  for root, dirs, files in os.walk(PATH_FILE, True):
      print("root:%s"% root)
      print("dirs:%s"% dirs)
      print("files:%s"% files)
      print("-------------------------------")

  for dirpath, subdirs, files in os.walk(PATH_FILE):
      for x in files:
          if x.endswith(dcm):
              dcmfiles.append(os.path.join(dirpath, x))

  dcmfiles = sorted(dcmfiles)
  
  for i in range(len(dcmfiles)):
    dcmfilessize.append(os.path.getsize(dcmfiles[i]))

  Dataset = pd.DataFrame({'Name':dcmfiles, 'Size':dcmfilessize, 'Arr':Arr}) 

  print(Dataset)

  total = len(dcmfilessize)

  for i in range(0, total, 2):

    print(dcmfilessize[i], '----', dcmfilessize[i + 1])

    if dcmfilessize[i] > dcmfilessize[i + 1]:
        Dataset.drop([i], axis = 0, inplace = True)
    else:
        Dataset.drop([i + 1], axis = 0, inplace = True)

  print(len(dcmfiles))
  print(len(dcmfilessize))
  print(len(Arr))

  print(Dataset)

  Paths = Dataset.iloc[:, 0].values
  Arr = Dataset.iloc[:, 2].values

  XsizeResized = 224
  YsizeResized = 224

  interpolation = cv2.INTER_CUBIC

  dsize = (XsizeResized, YsizeResized)

  Dataset.to_csv('Dataset' + NAME + '.csv')

  File = 0

  for File in range(len(Dataset)):

      ds = pydicom.dcmread(Paths[File])
      
      new_image = ds.pixel_array.astype(float)

      rescaled_image = (np.maximum(new_image, 0) / new_image.max()) * 255.0
      final_image = np.float64(rescaled_image)
      norm_img = np.zeros((224, 224))
      Resized_Imagen = cv2.resize(final_image, dsize, interpolation = interpolation)
      Resized_Imagen_final = cv2.normalize(Resized_Imagen,  norm_img, 0, 255, cv2.NORM_MINMAX)
      Name_File = Arr[File]

      dst_T = Name_File + '.png'
      dstPath_T = os.path.join(PathN, dst_T)
      dstPath_T1 = os.path.join(PathRe, dst_T)
      dstPath_T2 = os.path.join(PathReNo, dst_T)

      cv2.imwrite(dstPath_T, final_image)
      cv2.imwrite(dstPath_T1, Resized_Imagen)
      cv2.imwrite(dstPath_T2, Resized_Imagen_final)
      print('Images: ', Paths[File], '------', Arr[File])

# Convertion severity to int value

def CSV_Calc_DDSM(CSV_Path, Column, CalcOrMass):

    if CalcOrMass == 1:
        col_list = ["patient_id", "breast density", "left or right breast", "image view", "abnormality id", "abnormality type", "calc type", "calc distribution", "assessment", "pathology", "subtlety", "image file path", "cropped image file path", "ROI mask file path"]
    elif CalcOrMass == 2:
        col_list = ["patient_id", "breast_density", "left or right breast", "image view", "abnormality id", "abnormality type", "mass shape", "mass margins", "assessment", "pathology", "subtlety", "image file path", "cropped image file path", "ROI mask file path"]
    
    Dataset_Calc = pd.read_csv(CSV_Path, usecols = col_list)

    Dataset_Calc_Pathology, Dataset = TransformedLabelDDSM(Dataset_Calc, Column)

    print(Dataset_Calc_Pathology)
    pd.set_option('display.max_rows', Dataset.shape[0] + 1)
    print(Dataset.value_counts())

    return Dataset_Calc_Pathology

#

def ConfigurationModels(MainKeys, Arguments, MultiDataModels, MultiDataModelsEsp):

    TotalImage = []
    TotalLabel = []

    ClassSize = (len(Arguments[2]))
    Images = 7
    Labels = 8

    if len(Arguments) == len(MainKeys):
        
        DicAruments = dict(zip(MainKeys, Arguments))

        for i in range(ClassSize):

            for element in list(DicAruments.values())[Images]:
                TotalImage.append(element)
            
            #print('Total:', len(TotalImage))
        
            for element in list(DicAruments.values())[Labels]:
                TotalLabel.append(element)

            #print('Total:', len(TotalLabel))

            Images += 2
            Labels += 2

        #TotalImage = [*list(DicAruments.values())[Images], *list(DicAruments.values())[Images + 2]]
        
    elif len(Arguments) > len(MainKeys):

        TotalArguments = len(Arguments) - len(MainKeys)

        for i in range(TotalArguments // 2):

            MainKeys.append('Images ' + str(i + 3))
            MainKeys.append('Labels ' + str(i + 3))

        DicAruments = dict(zip(MainKeys, Arguments))

        for i in range(ClassSize):

            for element in list(DicAruments.values())[Images]:
                TotalImage.append(element)
            
            for element in list(DicAruments.values())[Labels]:
                TotalLabel.append(element)

            Images += 2
            Labels += 2

    elif len(Arguments) < len(MainKeys):

        raise ValueError('No se puede xD')

    #print(DicAruments)

    def printDict(DicAruments):

        for i in range(7):
            print(list(DicAruments.items())[i])

    printDict(DicAruments)

    print(len(TotalImage))
    print(len(TotalLabel))

    X_train, X_test, y_train, y_test = train_test_split(np.array(TotalImage), np.array(TotalLabel), test_size = 0.20, random_state = 3, shuffle = True)

    Score = PreTrainedModels(Arguments[0], Arguments[1], Arguments[2], Arguments[3], Arguments[4], ClassSize, Arguments[5], Arguments[6], X_train, y_train, X_test, y_test, MultiDataModels, MultiDataModelsEsp)
    #Score = PreTrainedModels(ModelPreTrained, technique, labels, Xsize, Ysize, num_classes, vali_split, epochs, X_train, y_train, X_test, y_test)
    return Score

#

def UpdateCSV(Score, df, column_names, path, row):

    print(df.head(len(df.index)))

    for i in range(len(Score)):
        df.loc[row, column_names[i]] = Score[i]
    
    print(df.head(len(df.index)))

    df.to_csv(path, index = False)
  
    print(df)