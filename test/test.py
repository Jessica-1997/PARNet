from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import numpy.matlib
import glob
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import re
import sys
sys.path.append('../')
from util.patches import OpenDataFiles2,OpenDataFilesLandsat
from loss  import SRE, UIQ,RMSE,PSNR
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
from collections import OrderedDict
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

global model

model_name='PARNet'


def SRNet(p10, p20): 
    p10 = np.expand_dims(p10, axis=0)
    p20 = np.expand_dims(p20, axis=0)    
    test = [p10, p20]     
    prediction = model.predict(test, verbose=1)    
    images = np.squeeze(prediction, axis=0)
    return images

def SRNet30(p10, p20, p30): 
    p10 = np.expand_dims(p10, axis=0)
    p20 = np.expand_dims(p20, axis=0)
    p30 = np.expand_dims(p30, axis=0)
    test = [p10, p20, p30]     
    prediction = model.predict(test, verbose=1)    
    images = np.squeeze(prediction, axis=0)
    return images


def Get_metrics(filepath, test_data, test_label):
    num_val = test_label.shape[0]
    num_band = test_label.shape[3]
    per_rmse = np.matlib.zeros((num_val, num_band))
    per_uiq = np.matlib.zeros((num_val, num_band))
    per_sre = np.matlib.zeros((num_val, num_band))
    per_psnr = np.matlib.zeros((num_val, num_band))
    # SR20 = []
    i=0     
    if len(test_data)==3:
        for img10, img20, img60, Val_img in zip(test_data[0], test_data[1], test_data[2], test_label):
            SR_img = SRNet30(img10, img20, img60)
            # np.save(filepath + '_60_'+model_name, SR_img)
            # SR20.append(SR_img)
            for band in range(num_band):
                x1 = SR_img[:, :, band]
                x2 = Val_img[:, :, band]
                per_rmse[i, band] = RMSE(x1, x2)
                per_uiq[i, band] = UIQ(x1, x2)
                per_sre[i, band] = SRE(x1, x2)
                per_psnr[i, band] = PSNR(x1, x2)
            i=i+1

    elif len(test_data)==2:
        for img10, img20, Val_img in zip(test_data[0], test_data[1], test_label):
            SR_img = SRNet(img10, img20)
            # np.save(filepath + '_'+model_name, SR_img)
            # SR20.append(SR_img)
            for band in range(num_band):
                x1 = SR_img[:, :, band]
                x2 = Val_img[:, :, band]
                per_rmse[i, band] = RMSE(x1, x2)
                per_uiq[i, band] = UIQ(x1, x2)
                per_sre[i, band] = SRE(x1, x2)
                per_psnr[i, band] = PSNR(x1, x2)
            i=i+1
    print("--- Done for all ---")
    del x1,x2
    rmse = per_rmse.mean(axis=0)
    uiq = per_uiq.mean(axis=0)
    sre = per_sre.mean(axis=0)
    psnr = per_psnr.mean(axis=0)   
    del per_rmse,per_uiq,per_sre,per_psnr
    
    return [rmse,uiq, sre, psnr]

def Get_metrics_Bic(test_data, test_label):
    num_val = test_label.shape[0]
    num_band = test_label.shape[3]
    per_rmse = np.matlib.zeros((num_val, num_band))
    per_uiq = np.matlib.zeros((num_val, num_band))
    per_sre = np.matlib.zeros((num_val, num_band))
    per_psnr = np.matlib.zeros((num_val, num_band))
    SR20 = []
    i=0    
    if len(test_data)==3:
        for bic_img, tru_img in zip(test_data[2], test_label):   
            for band in range(num_band):
                x1 = bic_img[:, :, band]
                x2 = tru_img[:, :, band]
                per_rmse[i, band] = RMSE(x1, x2)
                per_uiq[i, band] = UIQ(x1, x2)
                per_sre[i, band] = SRE(x1, x2)
                per_psnr[i, band] = PSNR(x1, x2)
            i=i+1
    elif len(test_data)==2:       
        for bic_img, tru_img in zip(test_data[1], test_label):   
            for band in range(num_band):
                x1 = bic_img[:, :, band]
                x2 = tru_img[:, :, band]
                per_rmse[i, band] = RMSE(x1, x2)
                per_uiq[i, band] = UIQ(x1, x2)
                per_sre[i, band] = SRE(x1, x2)
                per_psnr[i, band] = PSNR(x1, x2)
            i=i+1
    else:
        for bic_img, tru_img in zip(test_data, test_label):   
            for band in range(num_band):
                x1 = bic_img[:, :, band]
                x2 = tru_img[:, :, band]
                per_rmse[i, band] = RMSE(x1, x2)
                per_uiq[i, band] = UIQ(x1, x2)
                per_sre[i, band] = SRE(x1, x2)
                per_psnr[i, band] = PSNR(x1, x2)
            i=i+1
    print("--- Done for all ---")
    del x1,x2
    rmse = per_rmse.mean(axis=0)
    uiq = per_uiq.mean(axis=0)
    sre = per_sre.mean(axis=0)
    psnr = per_psnr.mean(axis=0)   
    del per_rmse,per_uiq,per_sre,per_psnr   
    return [rmse,uiq, sre, psnr]

# def SRNet( p20): 

#     p20 = np.expand_dims(p20, axis=0)    
#     test = p20   
#     prediction = model.predict(test, verbose=1)    
#     images = np.squeeze(prediction, axis=0)
#     return images

if __name__ == '__main__':

    run_Landsat = False

    rmse_list=[]
    uiq_list=[]
    sre_list=[]
    psnr_list=[]
    

                
    if run_Landsat:
        filelist=glob.glob('../data/train30/*_T1')

        for file in filelist:

            mode_id=file[26:32]
            model_file = model_name + '2_200_L11__'+mode_id+'__10_lr_1e-04.hdf5'
            MDL_PATH='../models/'+model_name+'/'+model_file 
            model=tf.keras.models.load_model(MDL_PATH)
            
            test_data, test_label = OpenDataFilesLandsat(file, Test = True)
            [rmse,uiq, sre, psnr] = Get_metrics(file, test_data, test_label)
            
            
            # test_data = np.load(file + '/Bicubic.npy')
            # test_label = np.load(file + '/data30_gt.npy')
            # test_ind = np.load(file + '/testing_index.npy')
            # test_label = test_label[test_ind]
            # test_label = test_label.transpose(0, 2, 3, 1)  
            # [rmse,uiq, sre, psnr] = Get_metrics_Bic(test_data, test_label)
            
            rmse_list.append(rmse)
            uiq_list.append(uiq)
            sre_list.append(sre)
            psnr_list.append(psnr)
            print('Done !')
            
    else:
        filelist=glob.glob('../data/train/*.SAFE')

        for file in filelist:
            mode_id=file[-20:-5]
            model_file = model_name + '_200_S11_'+mode_id+'__10_lr_1e-04.hdf5'
            MDL_PATH='../models/'+model_name+'/'+model_file 
            model=tf.keras.models.load_model(MDL_PATH)

            test_data, test_label = OpenDataFiles2(file, run_60, Test = True)
            [rmse,uiq, sre, psnr] = Get_metrics(file, test_data, test_label)
            
            # test_data = np.load(file + '/Bicubic.npy')
            # test_label = np.load(file + '/data20_gt.npy')
            # test_ind = np.load(file + '/testing_index.npy')
            # test_label = test_label[test_ind]
            # test_label = test_label.transpose(0, 2, 3, 1)  
            # [rmse,uiq, sre, psnr] = Get_metrics_Bic(test_data, test_label)
            
            rmse_list.append(rmse)
            uiq_list.append(uiq)
            sre_list.append(sre)
            psnr_list.append(psnr)
            print('Done !')
    

    
  


