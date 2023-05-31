import datetime
import re
import glob
import tensorflow as tf
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, ReduceLROnPlateau
from util.patches import OpenDataFiles
from models.DSen2Net import s2model
from models.PARNet import PARNet
from models.ESRCNN import SRCNN_model
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



model_name='PARNet'
out_path = '../models/'+model_name+'/'

if not os.path.isdir(out_path):
    os.mkdir(out_path)
    
class PlotLosses(Callback):
    def __init__(self, model_nr, lr):
        self.model_nr = model_nr
        self.lr = lr

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.i = 0
        self.x = []
        self.filename = out_path + self.model_nr + '_lr_{:.1e}.txt'.format(self.lr)
        open(self.filename, 'w').close()

    def on_epoch_end(self, epoch, logs=None):
        import matplotlib.pyplot as plt
        plt.ioff()

        lr = float(K.get_value(self.model.optimizer.lr))
        # data = np.loadtxt("training.log", skiprows=1, delimiter=',')
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.x.append(self.i)
        self.i += 1
        try:
            with open(self.filename, 'a') as self.f:
                self.f.write('Finished epoch {:5d}: loss {:.3e}, valid: {:.3e}, lr: {:.1e}\n'
                              .format(epoch, logs.get('loss'), logs.get('val_loss'), lr))
            plt.clf()
            plt.plot(self.x[0:], self.losses[0:], label='loss')
            plt.plot(self.x[0:], self.val_losses[0:], label='val_loss')
            plt.legend()
            plt.xlabel('epochs')
            # plt.waitforbuttonpress(0)
            plt.savefig(out_path + self.model_nr + '_loss0.png')
        except IOError:
            print('Network drive unavailable.')
            print(datetime.datetime.now().time())    
if __name__ == '__main__':
    

    lr = 1e-4
       
    INPUT_SHAPE =((None, None, 4), (None, None, 6))
    channels=6
    # create model

    # model = s2model(input_shape=INPUT_SHAPE, out_channels = channels, num_layers=6, feature_size=128)
    model = PARNet(input_shape = INPUT_SHAPE, out_channels = channels, feature_size =64, nblocks =6)
    # model = SRCNN_model(input_shape = INPUT_SHAPE, out_channels = channels)
    batch_size = 10

    nadam = Nadam(lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)  
    model.compile(optimizer=nadam, loss='mean_absolute_error', metrics=['mean_squared_error'])
    print('Model compiled.')

    S2filelist=glob.glob('../data/train/*.SAFE')

    for S2 in S2filelist:
        mode_id=S2[-20:-5]
        model_nr = model_name + '_200_S11_'+mode_id+'_'
        filepath = out_path + model_nr + '_' +str(batch_size) + '_' +'lr_{:.0e}.hdf5'.format(lr)

        checkpoint = ModelCheckpoint(filepath,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=True,
                                      save_weights_only=False,
                                      mode='auto')
        plot_losses = PlotLosses(model_nr, lr)
        LRreducer = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.5,
                                      patience=1,
                                      verbose=1,
                                      epsilon=1e-8,
                                      cooldown=0,
                                      min_lr=1e-8)
        callbacks_list = [checkpoint, plot_losses, LRreducer]
        print('Loading the training data...')
        train, label, val_tr, val_lb = OpenDataFiles2(S2, run_60 , Test = False)
        print(S2+'  Training starts...')
        model.fit(
        x=train,
        y=label,
        batch_size=batch_size,
        epochs=200,
        verbose=1,
        callbacks=callbacks_list,
        validation_split=0.00,
        validation_data=(val_tr, val_lb),
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0
        )
        print('Done !')




    
