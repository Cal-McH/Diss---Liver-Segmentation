import os

from timeit import default_timer as timer

import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import cv2
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from statistics import mean

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from torch.nn.functional import one_hot
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics

from alive_progress import alive_bar

import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:200"

start = timer()                                                                 ## Starts a timer to time the training and validation time

def scale(slice):                                                               ## Defines function to window, level, and normalise pixel values
    min = -160                                                                  ## Window and level values for contrast
    max = 240
    slice[slice < min] = min                                                    ## Masks values above or below the max/min
    slice[slice > max] = max
    slice = (slice - min) / (max - min)                                         ## Scales between 0-1
    new_slice = np.array(slice)                                                 ## Converts to numpy
    return new_slice                                                            ## Returns the scaled slice

def to_rgb(slice):
    rgb = cv2.cvtColor(slice, cv2.COLOR_GRAY2RGB)
    return rgb

def load_image(num, CT=None, SEG=None):                                         ## Takes inputs of number of patients we want, and CT or SEG = True for if we want CT or SEG
    assert not (CT is None and SEG is None)                                     ## Checks that we have actually inputted either CT=True or SEG=True
    assert CT or SEG                                                            ## Checks again that we have selected one or the other
    assert not (CT and SEG)                                                     ## Checks that we havent selected both

    if CT:                                                                      ## If CT is selected
        train_scaled = []                                                                   
        with alive_bar(num, force_tty=True) as bar:                             ## Allows us to use a progress bar
            for filename in sorted(os.listdir('./data/train/'))[:num]:           ## Loop through indexes up to the chosen patient value
                data = sitk.ReadImage(f'./data/train/{filename}')      ## Reads the CT file with the number and .nii suffix
                array = sitk.GetArrayFromImage(data)                            ## Gets the image array from the patient sitk object
                for slice in array:                                             ## Loops through the slices in the array
                    slice = np.float32(slice)  
                    train_scaled.append(np.array(to_rgb(np.float32(scale(slice)))))       ## Appends each slice to the train list
                bar()                                                           ## Updates the progress bar after every patient
        train_scaled = np.array(train_scaled)                                                 
        print(f'Scaled files: {len(train_scaled)}, Min: {train_scaled.min()}, Max: {train_scaled.max()}.')
        return train_scaled                                                     ## Ends the function
        
    elif SEG:                                                                   ## Repeats the same but for the segmentation images
        train_labels_raw = []
        with alive_bar(num, force_tty=True) as bar:
            for filename in sorted(os.listdir('./data/train_labels/'))[:num]:   
                data = sitk.ReadImage(f'./data/train_labels/{filename}')
                array = sitk.GetArrayFromImage(data)
                for slice in array:
                    slice = np.array(slice)
                    train_labels_raw.append(slice)
                bar()
        train_labels_raw = np.array(train_labels_raw)
        print(len(train_labels_raw))
        return train_labels_raw

train = load_image(5, CT=True) ## Creates the training data with 5 patients

train_seg = load_image(5, SEG=True) ## Creates the segmentation data using 5 patients data

print(train.shape, train_seg.shape)

# # PreProcessing

# ### Splitting to Train and Val 80:20

print('Splitting Data')

X_train, X_val, y_train, y_val = train_test_split(train, train_seg, test_size=0.20, random_state=1) ## Splits data 80:20 for train:val for both slice and seg data

print(f'Training: Length - {len(X_train)}, {len(y_train)}, Val: {len(X_val)}, {len(y_val)}')    ## Prints the length of each, 1+2 and 3+4 should both add to the length shown in the extraction function

# ### One-Hot Encoding Segmentations

n_classes = 3 ## Creates n classes (we have background = 0, liver = 1, and tumout = 2 - so three classes)

print('Converting masks to One-hot. This may take a while (~30s)')

from keras.utils import to_categorical ## Imports the to_categorical function from keras. This creates one-hot encoded channles for each class i.e tumour = 0,0,1
y_train_hot = to_categorical(y_train, num_classes=n_classes) ## Runs the one-hot on the train data
y_val_hot = to_categorical(y_val, num_classes=n_classes) ## Runs the one-hot on the val data

del y_train, y_val, train, train_seg ## Deletes unused variables to save memory

print('One-hot Complete')

# ### Reshaping data to N,C,H,W

print('Reshaping Data')

y_train_hot = np.transpose(y_train_hot, (0, 2, 3, 1)) ## Transposes the data shape from N,H,W,C to N,C,H,W as used in Tensors
y_train_hot = np.transpose(y_train_hot, (0, 2, 3, 1)) ## Have to do this twice for some reason

y_val_hot = np.transpose(y_val_hot, (0, 2, 3, 1))
y_val_hot = np.transpose(y_val_hot, (0, 2, 3, 1))

print(y_train_hot.shape, y_val_hot.shape)

X_train = np.transpose(X_train, (0, 2, 3, 1))
X_train = np.transpose(X_train, (0, 2, 3, 1))

X_val = np.transpose(X_val, (0, 2, 3, 1))
X_val = np.transpose(X_val, (0, 2, 3, 1))

print(X_train.shape, X_val.shape)

# ### Converting to Tensors

X_train = torch.as_tensor(X_train) ## Converts numpy lists to tensors
X_val = torch.as_tensor(X_val)
y_train = torch.as_tensor(y_train_hot)
y_val = torch.as_tensor(y_val_hot)

del y_train_hot, y_val_hot ## Deletes unused varables

# ### Normalise the slice data using ImageNet Mean and STD

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights ## Imports a FCN ResNet50 model and weights. Mainly used for the preprocessing.

## ResNet50
preprocess_img = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1.transforms(resize_size=None) ## Creates preprocessing function for the slice data.

# ### Creating DataSet and DataLoader

print('Processing Data. Noramlising to ImageNet weights')

X_train = preprocess_img(X_train) ## Normalises the data using imageNet trained weights (from FCN ResNet50 model) for better performance
X_val = preprocess_img(X_val)

train_set = TensorDataset(X_train,y_train) ## Creates a dataset using the training data and trainnin segmentations
train_dataloader = DataLoader(train_set, batch_size=10, pin_memory=True) ## Creates a dataloader from the dataset. No transformations needed as we have already done this

val_set = TensorDataset(X_val,y_val) ## Creates a dataset from the validation data and segmentations
val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, pin_memory=True)

print('Dataloaders Complete')

# # Model Architecture

# ## VGG19

# HyperParamters

ENCODER = 'vgg19'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = n_classes
ACTIVATION = 'softmax' # could be None for logits or 'softmax2d' for multiclass segmentation

# create segmentation model with pretrained encoder
model = smp.Unet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=CLASSES, 
    activation=ACTIVATION,
)
# Set to True if training, False if just predicting
TRAINING = True

# Set num of epochs
EPOCHS = 30
DEVICE = device

# define loss function
# loss = smp.utils.losses.DiceLoss()
loss = smp.utils.losses.DiceLoss()

# define metrics
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

# define optimiser
optimiser = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.0005), ## Uses learning rate of 0.0005
])

train_epoch = smp.utils.train.TrainEpoch( ## defines training epoch
    model, ## Uses the model defined above
    loss=loss, ## Usess loss function defined
    metrics=metrics, ## Usess Dice metrics defined
    optimizer=optimiser, ## Usess optimiser defined
    device=DEVICE, ## Usess the device defined - hopefully should be CUDA
    verbose=True, ## Plots a progress bar
)

valid_epoch = smp.utils.train.ValidEpoch( ## Same as the training epoch but uses no optimiser as we are validaing here
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

print(f'Running {ENCODER} model')

if TRAINING: ## Training should be set to True above if we want to train

    best_iou_score = 0.0 ## Logs current best IoU
    train_logs_list, valid_logs_list = [], [] ## Createsa list for the training and validation logs

    for i in range(0, EPOCHS): ## Loops through epochs in range up to the epoch number specified earlier

        # Perform training & validation
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dataloader) ## Runs epoch 'n' using the train epoch and the train dataloader
        valid_logs = valid_epoch.run(val_dataloader) ## Runs val epoch 'n' using the val epoch and the val dataloader
        train_logs_list.append(train_logs) ## Appends logs to the train logs
        valid_logs_list.append(valid_logs) ## Appends val logs to the val logs

        if best_iou_score < valid_logs['iou_score']: ## Determines if the new IoU is better than the previous
            best_iou_score = valid_logs['iou_score'] ## If so, update the IoU score
            torch.save(model, f'./VGG19_n5_e30_Focal.pth') ## Save the new model
            print('Model saved!')

print('Model Training Complete')

train_logs_df = pd.DataFrame(train_logs_list) ## Converts logs to dataframes
valid_logs_df = pd.DataFrame(valid_logs_list)
print('Training IoU Logs')
train_logs_df.T

print('Validation IoU Logs')
valid_logs_df.T

train_logs_df.to_csv(r'./VGG19_train_n5_e30.csv', sep='\t', encoding='utf-8', header='true') ## Saves dataframes as CSVs
valid_logs_df.to_csv(r'./VGG19_val_n5_e30.csv', sep='\t', encoding='utf-8', header='true')

print('Exported Logs as CSV')

print('Plotting IoU over Epoch')

plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.iou_score.tolist(), lw=3, label = 'Train') ## Plots train logs IoU scores
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.iou_score.tolist(), lw=3, label = 'Valid') ## Plots val logs IoU scores
plt.xlabel('Epochs', fontsize=21)
plt.ylabel('IoU Score', fontsize=21)
plt.title('VGG19 IoU Score over Epoch', fontsize=21)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig('VGG19_n5_e30_IoU.png')
plt.show()

print('Plotting Dice Loss over Epoch')

plt.figure(figsize=(20,8))
plt.plot(train_logs_df.index.tolist(), train_logs_df.dice_loss.tolist(), lw=3, label = 'Train') ## Plots train logs Dice scores
plt.plot(valid_logs_df.index.tolist(), valid_logs_df.dice_loss.tolist(), lw=3, label = 'Valid') ## Plots val logs Dice scores
plt.xlabel('Epochs', fontsize=21)
plt.ylabel('Dice Loss', fontsize=21)
plt.title('VGG19 Dice Loss Plot over Epoch', fontsize=21)
plt.legend(loc='best', fontsize=16)
plt.grid()
plt.savefig('VGG19_n5_e30_Dice.png')
plt.show()

end = timer()

time_elapsed = (end - start)

print(end - start)

print(time_elapsed)


