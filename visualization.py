from scipy import io as sio
from scipy.ndimage import rotate
from scipy.misc import imsave
from viztools import dice_coef, burn_line, loadnii, makergb,create_mosaic,construct_colormap,overlay_colormap,prepare_for_rgb
from PIL import ImageDraw
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

###################### here's where the main begins ###########################
alpha = 0.5 # color hue
display_slice = 16
shortened_slice = display_slice - 13

################################### specifying directories ###################################
Anthony_dir = '/Users/yuanxie/ASLpredictTmax/mrCNNfinal/' # On longo:'/data3/yuanxie/project_stroke_mask/'
# Yuan_dir = '/Users/yuanxie/ASLpredictTmax/mrUNetfinal/'
Yuan_dir = '/Users/yuanxie/ASLpredictTmax/mrUNetfinal/'
Source_dir = '/Users/yuanxie/ASLpredictTmax/ASL2PWI_107_full/'
Input_dir = '/Users/yuanxie/ASLpredictTmax/preprocess_asl2pwi_may/'
Output_dir = '/Users/yuanxie/ASLpredictTmax/Figures/'
# subj_id = '30008BL'
# subj_id = '30122BL'

# create T1 mask!
T1t = loadnii('/Users/yuanxie/controls_stroke_DL/001/T1.nii')
img_T1mask = T1t > 0 * 1.
# weee!

for subj_id in ["30001BL"
,"30002BL"
,"30006BL"
,"30007BL"
,"30008BL"
,"30010FU1"
,"30012BL"
,"30018BL"
,"30019FU1"
,"30023BL"
,"30024BL"
,"30024FU1"
,"30025FU1"
,"30026BL"
,"30027BL"
,"30027FU1"
,"30030BL"
,"30030FU1"
,"30032BL"
,"30033FU1"
,"30035BL"
,"30035FU2"
,"30036FU1"
,"30037BL"
,"30037FU1"
,"30039FU2"
,"30040BL"
,"30041BL"
,"30042BL"
,"30043BL"
,"30043FU1"
,"30044FU2"
,"30045BL"
,"30047BL"
,"30047FU1"
,"30047FU2"
,"30048BL"
,"30049BL"
,"30049FU1"
,"30051FU1"
,"30051FU2"
,"30053BL"
,"30055BL"
,"30056BL"
,"30057BL"
,"30058BL"
,"30059BL"
,"30063BL"
,"30069BL"
,"30072FU1"
,"30073BL"
,"30073FU1"
,"30075BL"
,"30077BL"
,"30078BL"
,"30080BL"
,"30081FU1"
,"30082BL"
,"30084BL"
,"30086FU1"
,"30092BL"
,"30096BL"
,"30097BL"
,"30098BL"
,"30099BL"
,"30100BL"
,"30101BL"
,"30103BL"
,"30104BL"
,"30106BL"
,"30108BL"
,"30109BL"
,"30110BL"
,"30111FU1"
,"30115BL"
,"30115FU1"
,"30116BL"
,"30117BL"
,"30120BL"
,"30122BL"
,"30124BL"
,"30126BL"
,"30127BL"
,"30129BL"
,"30130BL"
,"30130FU1"
,"30131BL"
,"30134BL"
,"30135BL"
,"30137BL"
,"30138BL"
,"30141BL"
,"30146BL"
,"30147FU2"
,"30148BL"
,"30152BL"
,"30153BL"
,"30154BL"
,"30155BL"
,"30155FU1"
,"30156BL"
,"30157BL"
,"30158BL"
,"30160BL"
,"30161BL"]:
	print('handling patient ', subj_id)
	yuan_prediction = Yuan_dir + 'prediction_' + subj_id + '.nii'
	# yuan_predictionnpy = Yuan_dir + 'predictnpy_' + subj_id + '.npy'
	yuan_groundtruth = Yuan_dir + 'gt_' + subj_id + '.nii'
	yuan_background = Yuan_dir + 'flair_' + subj_id + '.nii'

	database_groundtruth = Source_dir + subj_id + '/wTmax_seg.nii'
	database_background = Source_dir + subj_id + '/wDWI.nii'
	database_tmax = Source_dir + subj_id + '/wtmax.nii'
	database_adc = Source_dir + subj_id + '/wADC.nii'
	database_aslcbf = Source_dir + subj_id + '/wrASLCBF.nii'
	# inputs
	input_sequences = Input_dir + subj_id + '/inputs_aug0.hdf5'
	input_gt = Input_dir + subj_id + '/output_aug0.hdf5'


	################################### loading images ###################################
	# load yuan's
	# img_yuan_predictionnpy = np.load(yuan_predictionnpy)
	img_yuan_prediction = loadnii(yuan_prediction)
	print('yuan_pred_nii:',img_yuan_prediction.shape)
	img_yuan_groundtruth = loadnii(yuan_groundtruth)
	img_yuan_background = loadnii(yuan_background)
	print('yuan:',img_yuan_prediction.shape, img_yuan_groundtruth.shape, img_yuan_background.shape)

	# load the raw data source (sanity check due to suspicious activities)
	img_database_groundtruth = loadnii(database_groundtruth)
	img_database_background = loadnii(database_background)
	img_database_tmax = loadnii(database_tmax)
	img_database_adc = loadnii(database_adc)
	img_database_aslcbf = loadnii(database_aslcbf)

	print('database:',img_database_groundtruth.shape, img_database_background.shape)

	# preprocessed inputs - also load sequences for visualization
	img_input_sequences = h5py.File(input_sequences, 'r')['init']
	img_input_dwi = np.squeeze(img_input_sequences[:,:,:,0])

	img_input_adc = np.squeeze(img_input_sequences[:,:,:,1])
	img_input_aslcbf = np.squeeze(img_input_sequences[:,:,:,2])
	img_input_dwi = img_input_dwi.transpose(1,2,0)
	img_input_adc = img_input_adc.transpose(1,2,0)
	img_input_aslcbf = img_input_aslcbf.transpose(1,2,0)

	# dwi mask based on 91,109,91
	mean_dwi = np.mean(img_database_background[np.nonzero(img_database_background)])
	img_dwimask = img_database_background > (0.3 * mean_dwi)
	# ground truth
	img_input_gt = h5py.File(input_gt, 'r')['init']
	img_input_gt = np.squeeze(img_input_gt)
	img_input_gt = img_input_gt.transpose(1,2,0)

	print('inputs:',img_input_gt.shape,img_input_dwi.shape)

	# here we start 
	pad = 13 # all images from yuan pred will be slice - pad (cuz of cropping)
	rows = 91
	cols = 109
	thres = 0.5
	list_img_masked = []
	list_img = []

	img_database_tmax[np.isnan(img_database_tmax)] = 0
	img_database_tmax[img_database_tmax < 0] = 0

	[xx,yy,predictionslice]= img_yuan_prediction.shape
	dicescore = dice_coef(img_database_groundtruth[:,:,pad:pad+predictionslice],img_yuan_prediction > 0.5)
	# tmax modification ends
	for s in range(13,69):
		# extract background, ground truth, mrcnn and unet prediction slice
		# in each loop, pick slice s of image and plot the DWI,ADC,ASLCBF,TMAX (not an input), UNET prediction

		img_gt = img_database_groundtruth[:,:,s]
		T1mask = img_T1mask[:,:,s]
		img_pred_unet = img_yuan_prediction[:,:,s - pad] # 91, 109
		
		# input sequence loading
		dwimask = np.rot90(np.squeeze(img_dwimask[:,:,s]),1) * np.rot90(T1mask,1)
		##
		# plt.imshow(dwimask)
		# plt.show()
		## 
		img_dwi = np.rot90(np.squeeze(img_database_background[:,:,s]),1) * dwimask
		img_adc = np.rot90(np.squeeze(img_database_adc[:,:,s]),1) * dwimask
		img_aslcbf = np.rot90(np.squeeze(img_database_aslcbf[:,0:109,s]),1) * dwimask
		img_tmax = np.rot90(np.squeeze(img_database_tmax[:,:,s]),1) * dwimask
		# img_tmax = cv2.resize(img_tmax, dsize=(128,128), interpolation=cv2.INTER_LINEAR) * dwimask
		[img_dwi,img_adc,img_aslcbf,img_tmax] = prepare_for_rgb(img_dwi,[img_adc,img_aslcbf,img_tmax])
		img_background = img_dwi
		img_dwi_color = makergb(img_dwi)
		img_adc_color = makergb(img_adc)
		img_aslcbf_color = makergb(img_aslcbf)
		img_tmax_color = makergb(img_tmax)

		# threshould prediction maps
		img_pred_unet = 1.0*(img_pred_unet > 0.5)

		# Construct a color mask for each prediction to superimpose

		color_mask_unet = construct_colormap(rows,cols,img_gt,img_pred_unet)
		color_mask_unet = np.rot90(color_mask_unet,1)
		# make overlayed colormap

		img_background_unet_overlayed = overlay_colormap(img_background, color_mask_unet, alpha)

		# sequence_figure = form_grid_ASL2PWI(img_pred_unet, img_gt, img_dwi, img_tmax, img_adc, img_aslcbf)
		sequence_figure = np.concatenate((img_dwi_color,img_adc_color,img_aslcbf_color, 
			img_tmax_color, img_background_unet_overlayed),axis = 1)
		# sanity check again
		# plt.imshow(sequence_figure)
		# plt.show()

		if not os.path.isdir(Output_dir + subj_id): 
			os.mkdir(Output_dir + subj_id)

		# dicomsave(Output_dir, subj_id, s, sequence_figure[:,:,0])
		imgname = Output_dir + subj_id + '/' + subj_id + '_{0:03}.png'.format(s)
		imsave(imgname, sequence_figure)
		# burn dice to the image?
		burn_line('dice: {0:.3f}'.format(dicescore), imgname)
