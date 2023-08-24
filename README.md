# Dissertation --- Liver-Segmentation

## UNET CNN Models for Liver and Tumour Segmentation in CRCLM Patients


# Data

The data is provided by the LiTS Codalab challenge, taking CT volume data from 130 patient with liver cancer. There are 70 patients in the testing 
folder, however these were not used in this project as they did not contain the segmentations for the required testing metrics. Data can be found at 
https://competitions.codalab.org/competitions/17094 .

# File Processing

If you would like to run the python scripts, please split the training data into sepparate training and testing with the file directories: 
- Training CT volumes: 'data/train/...' 
- Training Segmentation volumes: 'data/train_labels/...' 
- Testing CT volumes: 'data/test/...' 
- Testing segmentations: 'data/test_labels/...' 

Please make sure that you extract the corresponding patient numbers for both the CT volumes and the segmentation volumes so that the slices are alligned once extracted. If not, the  metrics will be calculated incorrectly.

# Running Models

If the files have been split correctly, you should be able to run the corresponding python script for the model you would like to run. By Default, the model only extracts the data from the first 5 patients. Depending how you sorted the files, this should be around 3000 images. If you would like to extract more, please change the value in the 'load_image()' function to a higher/lower value. This is left at 5 due to computational constraints on the machine I was using.

Whilst running the models, the code will automatically decide to send data in batches to the GPU if one is installed. If not, it will run on the CPU and will be extremely slow. 

On a GPU machine, 5 patients data takes around;

- ResNet50: 35 Minutes
- ResNet101: 1 Hour 15 Mins
- VGG19 : 1 Hour 35 Mins

# Testing Models

This is currently set up to what I did. If you would like to test on a single model, please just use the data extraction, preprocessing, and model testing epoch loop and run individually. The plotting gets a little more complex as parts of the script were run more than once / not in order. Not the best practice, but I did not intend for this to be run as a single script.

The testing if performed in a python Notebook to easily track the testing metrics. This doesn't take too long, however if you'd like to test on more patients then it will take longer. 
When testing, please make sure to edit the model names and saved CSV files if you have renamed these in the python scripts before.

When running the final plots, please make sure to only run these once you have run the tests 3 times on different datasets. The data extraction uses 'start' and 'end' values which denote the starting and ending index of files to be extracted from the testing folders. You should extract the range of files desired, preprocess, and run the testing epochs fully. Then, restart the kernel with a different dataset and change the name of the saved CSV files to represent the new dataset range. This should be run three times to get three testing metric CSVs from each of the three models.

If you'd like to only plot one dataset then please change the matplotlib subplots. 

The plots also have manually entered metrics from the testing CSVs. These will need to be changed manually to your own results. I have left in my test metrics as placeholders but these will need to be changed to your own values. These will be printed once you have uploaded all CSVs (which should be renamed), but the order inputted into the dataframe dictionary is not intuitive. In the plot dataframe, the values should be entered in the 'IoU: []' and 'Dice: []' as;

- Resnet50 Test 1 IoU/Dice
- Resnet101 Test 1 IoU/Dice
- VGG19 Test 1 IoU/Dice
- Resnet50 Test 2 IoU/Dice
- Resnet101 Test 2 IoU/Dice
- VGG19 Test 2 IoU/Dice
- Resnet50 Test 3 IoU/Dice
- Resnet101 Test 3 IoU/Dice
- VGG19 Test 3 IoU/Dice

Not very intuitive but it works. Just a bit confusing when entering your own values. Ideally should be changed to extract the correct index from the nine CSVs.
