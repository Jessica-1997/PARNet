from random import randrange
import random
import numpy as np
import glob
import sys
import os
os.getcwd()



def Train_Vali_Test(filepath):

    data10 = np.load(filepath + '/data10.npy')
    size= data10.shape[0]
    ratio = .1
    train_index = []
    vali_index = []
    test_index = []
    index_range = [i for i in range(size)]  
    test_index = random.sample(index_range, int(size * ratio))
    for test in test_index:
        index_range.remove(test)
        
    vali_index= random.sample(index_range, int(size * ratio))
    for vali in vali_index:
        index_range.remove(vali)
        
    train_index = index_range

    np.save(filepath + '/training_index.npy', train_index)
    np.save(filepath + '/validation_index.npy', vali_index)
    np.save(filepath + '/testing_index.npy', test_index)
    
    Tol_patches = train_index.__len__() + vali_index.__len__() + test_index.__len__()
    print('Full no of samples: {}'.format(Tol_patches))
    print('Training samples: {}'.format(train_index.__len__()))
    print('Validation samples: {}'.format(vali_index.__len__()))
    print('Testing samples: {}'.format(test_index.__len__()))
    

run_Landsat =False

if run_Landsat:
    filelist=glob.glob('../data/train30/*_T1')
else:
    filelist=glob.glob('../data/train/*.SAFE')


filelist = ['../data/train/S2B_MSIL1C_20201024T030809_N0209_R075_T50TMK_20201024T051752.SAFE'] 
for file in filelist:  
    Train_Vali_Test(file)





