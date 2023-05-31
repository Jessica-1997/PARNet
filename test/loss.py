import tensorflow as tf
import numpy as np



def log10(x):
    """
    Compute log base 10
    """
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def l1_loss(y_true, y_pred, y_mask, HR_SIZE):
    """
    Modified l1 loss to take into account pixel shifts
    """
    return 0

def ssim(y_true, y_pred, y_mask, size_image, clear_only=False):
    """
    Modified SSIM metric to take into account pixel shifts
    """ 
    return 0


def RMSE(x1, x2):
    
    diff = x1.astype(np.float64)-x2.astype(np.float64)
    rms = np.sqrt(np.mean(np.power(diff, 2)))
    # print('RMSE: {:.4f}'.format(rms))
    return rms 

    
def SRE(pred_img, org_img):   
    
    org_img = org_img.astype(np.float64)
    pred_img=pred_img.astype(np.float64)
    
    numerator = np.square(np.mean(org_img))
    n=org_img.shape[0] * org_img.shape[1]
    denominator = (np.linalg.norm(pred_img - org_img, 2))                                     
    sre = 10 * np.log10(n*numerator/denominator)
    # print('SRE: {:.4f}'.format(sre))   
    return sre

def UIQ(pred_img, org_img):
    org_img = org_img.astype(np.float64)
    pred_img=pred_img.astype(np.float64)
    
    n=pred_img.shape[0]*pred_img.shape[1]
    pred_img=pred_img.reshape(1,n)
    org_img=org_img.reshape(1,n)
    
    org_mean=np.mean(org_img)    
    pred_mean=np.mean(pred_img)
    
    Cov_org_pred=np.cov(org_img,pred_img)
    cov_org2=Cov_org_pred[0][0]
    cov_pred2=Cov_org_pred[1][1]
    cov_org_pred=Cov_org_pred[0][1]
    
    down=(cov_org2+cov_pred2)*(np.power(org_mean, 2)+np.power(pred_mean, 2))
    Q=4*cov_org_pred*org_mean*pred_mean/down
    # print('UIQ: {:.4f}'.format(Q))
    
    return Q

def PSNR(pred_img, org_img):
    #图像缩放在0~255之间
    rmse=RMSE(pred_img, org_img)
    max_I=org_img.max()
    _psnr=20* np.log(max_I.astype(np.float64)/rmse)
    # print('PSNR: {:.4f}'.format(_psnr))
    return _psnr









































