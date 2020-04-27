from scipy import io as sio
from scipy.ndimage import rotate
from scipy.misc import imsave
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import pydicom as pd
from pydicom.data import get_testdata_files
import nibabel as nib
import sys
from glob import glob
from skimage import data, color, io, img_as_float
import matplotlib.pyplot as plt
import h5py
import cv2
import itertools
import math
import datetime
import csv
# from keras import backend as K
# from sklearn.metrics import roc_curve, auc, confusion_matrix
def whichcategory(pred,gt):
	true = [1, '1', 'TRUE', True]
	false = [0, '0', 'FALSE', False]
	if pred in true and gt in true:
		return 'TP'
	elif pred in true and gt in false:
		return 'FP'
	elif pred in false and gt in true:
		return 'FN'
	elif pred in false and gt in false:
		return 'TN'
	else:
		return 'N/A'


def readcsv(filepath):
	with open(filepath, mode='r') as infile:
		reader = csv.reader(infile)
		mydict = {rows[0]:rows[1:] for rows in reader}

	return mydict

def Triage(penumbra, lesion):
	return (lesion < 70 and penumbra > 1.8 * lesion and penumbra - lesion > 15) - 0 
	# return lesion < 70 and penumbra < 100 and penumbra > 1.8 * lesion and penumbra - lesion > 15

def thresholdADCLesion(inputadc, thres = 620, voxelsize = 8/1000):
	if not len(inputadc.shape) == 3: 
		raise Exception('input should be 3D volume. The shape of input was: ', inputadc.shape)
	mask = loadnii('/Users/yuanxie/controls_stroke_DL/001/template.nii')
	mask_tailored = mask[:,:,13:73] > 0.1
	volume = mask_tailored * (inputadc < thres)
	return volume, sum(volume.flatten()) * voxelsize

def burn_line(inputstring, targetfig, x = 1, y = 1, color = 'rgb(255,255,255)'):
	image = Image.open(targetfig)
	draw = ImageDraw.Draw(image)
	font = ImageFont.truetype('Arial', size=12)
	draw.text((x, y), inputstring, fill = color, font = font)
	image.save(targetfig)

def writecsv(inputlist,filename):
	with open(filename, "w") as csv_file:
		writer = csv.writer(csv_file, delimiter=',')
		for line in inputlist:
			writer.writerow(line)

def append_uid(input_str,suffix=''):
	output_str = '1.2.840.1'+input_str+suffix
	return output_str

def youden_thres(fpr,tpr,thresholds,verbose = False):
		# plot youden's index over threshold range
	youden = tpr - fpr
	i = np.argmax(youden)
	sens = tpr[i]; spec = 1-fpr[i]; F1 = thresholds[i];
	if verbose:
		print('sensitivity with maximized youden index is',sens)
		print('specificity with maximized youden index is',spec)
		print('threshold is', F1)
		plt.plot(youden)
		plt.title('youden index curve, max at threshold {0:.3f}'.format(F1))
		plt.show()

	return F1

def dice_coef(y_true, y_pred, smooth=1):
	y_true_f = y_true.flatten()
	y_pred_f = y_pred.flatten()
	intersection = sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)

def loadnii(path):
	data = nib.load(path)
	output = data.get_fdata()
	output = np.maximum(0, np.nan_to_num(output, 0))
	return output

def makergb(vol):
	vol = (vol - np.min(vol))/np.ptp(vol)
	return np.dstack((vol, vol, vol))

def create_mosaic(start_slice, end_slice, list_img, list_img_masked, row_num = 5, spacing = 2):
	'''
	takes in the starting slice #, end slice #, a list of original image and a list of masked image
	outputs the mosaic layout of original image, masked image in a fashion that 5 images are concatonated in a row
	in the img_both output, original images are put side-by-side to the masked images for comparison
	row_num - how many images we want in a row
	spacing - 1, append every image; 2, append every other image; so on so forth
	TODO: add automatic padding of zeros if the rest doesn't fill in a full 5 image row
	'''
	full_mosaic = np.array([])
	masked_mosaic = np.array([])
	for n in range(start_slice,end_slice,row_num*spacing):
		full_row = np.array([])
		masked_row = np.array([])
		for k in range(0,row_num*spacing, spacing):
			
			if (full_row.size == 0):
				print('slice#: ', n+k, "------ a new row begins ---------")
				full_row = np.concatenate((list_img[n+k], list_img_masked[n+k]), axis = 1)
				masked_row = list_img_masked[n+k]
				print('fullrow_size: ', full_row.shape)
				print('maskedrow_size: ', masked_row.shape)
			else:
				print('slice#: ', n+k)
				print('fullrow_size: ', full_row.shape)
				print('maskedrow_size: ', masked_row.shape)
				full_row = np.concatenate((full_row, list_img[n+k], list_img_masked[n+k]), axis = 1)
				masked_row = np.concatenate((masked_row, list_img_masked[n+k]), axis = 1)
			
		# now append each row
		if (full_mosaic.size == 0):
			
			full_mosaic = full_row
			masked_mosaic = masked_row
			print("------ first row of image completes ---------")
			print('fullmosaic_size: ', full_mosaic.shape)
			print('maskedmosaic_size: ', masked_mosaic.shape)
		else:
			
			full_mosaic = np.concatenate((full_mosaic, full_row), axis = 0)
			masked_mosaic = np.concatenate((masked_mosaic, masked_row), axis = 0)
			print("------ another row of image completes ---------")
			print('fullmosaic_size: ', full_mosaic.shape)
			print('maskedmosaic_size: ', masked_mosaic.shape)

	return[full_mosaic, masked_mosaic]


def construct_colormap(rows,cols,ground_truth,prediction):
	'''
	- rows and cols are the input grount truth and prediction's size
	- ground_truth and prediction is assumed to be both 2D images
	- output color_mask is a 3D rgb numpy array
	'''
	color_mask = np.zeros((rows, cols, 3))
	for r in range(0,rows):
		for c in range(0,cols):
			# true positive, Green
			if (ground_truth[r,c] == 1 and prediction[r,c]==1): color_mask[r,c,:] = [0, 1, 0]  
			# false negative, Blue
			if (ground_truth[r,c] == 1 and prediction[r,c]==0): color_mask[r,c,:] = [0, 0, 1]  
			# false positive, Red + a little green = orange?????
			if (ground_truth[r,c] == 0 and prediction[r,c]==1): color_mask[r,c,:] = [1, 0.2, 0]  
	
	return color_mask

def overlay_colormap(img_background, color_mask, alpha):
	img_background_color = makergb(img_background)

	img_background_hsv = color.rgb2hsv(img_background_color)
	color_mask_hsv = color.rgb2hsv(color_mask)

	# Replace the hue and saturation of the original image
	# with that of the color mask - not sure what it means
	img_background_hsv[..., 0] = color_mask_hsv[..., 0]
	img_background_hsv[..., 1] = color_mask_hsv[..., 1] * alpha
	return color.hsv2rgb(img_background_hsv)

def prepare_for_rgb(leadvol,listvol):
	leadvol[leadvol < 0] = 0
	m_lead = np.mean(leadvol[leadvol > 0])
	reslist = [leadvol]
	# no normalization, bring up the intensity of tmax and aslcbf
	for vol in listvol:
		vol[vol < 0] = 0 
		m_vol = np.mean(vol[vol > 0])
		if not m_vol == 0:
			vol = vol * m_lead / m_vol
			vol[vol > 5 * m_lead] = 4 * m_lead
		reslist.append(vol)
	return reslist

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          labelx = '',
                          labely = '',
                          sourcedir = '.'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.ylabel(labely)
    plt.xlabel(labelx)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.savefig(os.path.join(sourcedir,'triage_confm.png'))

def sanitycheck(inputvol, targetslice = 0, titleline = 'sanitycheck'):
	'''
	inputvol.shape = [row,column,slices,number of volumes]
	'''
	row,col,z,nvols = inputvol.shape
	f,ax = plt.subplots(1, nvols, subplot_kw={'xticks': [], 'yticks': []})
	for i in range(nvols):
		ax[i].imshow(np.squeeze(inputvol[:,:,targetslice,i]), cmap=plt.cm.gray)
	plt.title(titleline)
	plt.show()
	
'''
Here are the failed functions

def dicomsave_alsorubbish(Output_dir,subj_id,slicenumber,image):	
	print("Saving", subj_id, 'slice',slicenumber)
	template = get_testdata_files('MR_small.dcm')[0]
	ds = pd.dcmread(template)
	filename= os.path.join(Output_dir,subj_id,subj_id + '_{0:03}.dcm'.format(slicenumber))
	# copy the data back to the original data set
	ds.PixelData = image.tostring()
	# update the information regarding the shape of the data array
	ds.Rows, ds.Columns = image.shape
	ds.PatientName = subj_id
	ds.PatientID = subj_id
	ds.SliceLocation = slicenumber * 5
	ds.ImagePositionPatient = [0,0,slicenumber*5]

	dt = datetime.datetime.now()
	timeStr = dt.strftime('%Y%m%d%H%M%S.%f')
	ds.SeriesInstanceUID = append_uid(timeStr)
	ds.SOPInstanceUID = append_uid(timeStr)
	ds.save_as(filename)

def dicomsave_rubbish(Output_dir,subj_id,slicenumber,image,template = '/Users/yuanxie/controls_stroke_DL/001/template.dcm'):
	# default template
	ds_template = pd.read_file(template)
	# Create filenames
	filename= os.path.join(Output_dir,subj_id,subj_id + '_{0:03}.dcm'.format(slicenumber))

	print("Setting file meta information...")
	# Populate required values for file meta information
	file_meta = Dataset()
	file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
	file_meta.MediaStorageSOPInstanceUID = "1.2.3"
	file_meta.ImplementationClassUID = "1.2.3.4"

	print("Setting dataset values...")
	# Create the FileDataset instance (initially no data elements, but file_meta
	# supplied)
	ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

	# Add the data elements -- not trying to set all required here. Check DICOM
	# standard
	ds.PatientName = subj_id
	ds.PatientID = subj_id

	# Set the transfer syntax
	ds.is_little_endian = True
	ds.is_implicit_VR = True

	# Set creation date/time
	dt = datetime.datetime.now()
	ds.ContentDate = dt.strftime('%Y%m%d')
	timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
	ds.ContentTime = timeStr

	ds.pixel_array.setflags(write=1)
	ds.PatientName = 'Anonymous'
	ds.pixel_array[:,:] = image
	ds.PixelData = ds.pixel_array.tobytes()
	ds.Rows, ds.Columns = image.shape
	ds.SliceLocation = slicenumber * 5
	ds.ImagePositionPatient = [0,0,slicenumber*5]

	print("Writing test file", filename)
	ds.save_as(filename)


	###
	# dt = datetime.datetime.now()
	# timeStr = dt.strftime('%Y%m%d%H%M%S.%f')
	# ds = pd.read_file(template)

	ds.pixel_array.setflags(write=1)
	ds.PatientName = 'Anonymous'
	ds.pixel_array[:,:] = image

	ds.PixelData = ds.pixel_array.tobytes()
	ds.Rows, ds.Columns = image.shape
	# ds_template.SeriesInstanceUID = append_uid(timeStr)
	# ds_template.SOPInstanceUID = append_uid(timeStr,str(slicenumber))
	ds.SliceLocation = slicenumber * 5
	ds.ImagePositionPatient = [0,0,slicenumber*5]
	# ds_template.save_as(os.path.join(Output_dir,subj_id,subj_id + '_{0:03}.dcm'.format(slicenumber)))
'''
