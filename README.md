# CNN-CLASSIFIER #
This is a generic convolutional model which can train a set of images and predict a output of it.In this project I have collected a set of christmas object images and developed a model and trained it.Now from a picture it can predict whether it is realted to christmas or not.Since it is generic you can able to use this model for various scenarios.

## Python Library Dependencies ##
  
+ [Tensorflow](https://www.tensorflow.org/)   
+ [numpy](http://www.numpy.org/)

## LINK FOR THE TRAINING IMAGE SOURCE ##
https://mega.nz/#F!SxAHXRTZ!rnbP8APGdCqponSbg5C-pQ

**Core Functionality**
+ `preprocess.py` - Process the image dataset to the required format
+ `imageagument.py` - Agumenting the image to get improve the accuracy
+ `christmas.tfrecords` - Its a representation of image dataset in a well structured format 
+ `train.py` - Deploying and Training the model

# SAMPLE WORK #
## TRAINING IMAGES: ##
![alt text](https://www.cakengifts.in/product-images/bfcr001-black-forest-cake-in-round/regular/black-forest-cake-in-round.jpg "CAKE")
![alt text](https://houseandhome.com/wp-content/uploads/small-christmas-tree-ideas-bhg.jpg)
![alt text](http://janicelukes.ca/wp-content/uploads/2017/11/Skate-With-Santa-20121116-small.jpg)
![alt text](https://www.bikeinflorence.com/wp-content/uploads/2015/12/albero-di-natale-caminetto.jpeg)

## TEST AND OUTPUT ##
![alt text](https://www.lds.org/bc/content/ldsorg/content/images/2011mtc-christmas-480x270-CWD_100705_KMIller_TempleSquareLights_04_038.jpg)

**OUTPUT : CHRISTMAS IMAGE**

**ACCURACY OF THE MODEL: 72.2%** 


