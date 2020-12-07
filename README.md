# EVA5_MidasYoloPlanarcnn

## Description 
How amazing it is when a camera can make up what is in front of it, how many of it, and possibly take actions accordingly just the way we do. So this assignment talks about one such revolutionary algorithm that has not only made all this possible but that too with brilliant precision - YOLO Algorithm. Yolo which stands for - You Only Look Once, is an "**Object Detection**" algorithm that deals with detecting instances of various classes of objects like person, cat, dog, book, tie, car, etc. We'll discuss the working in detail as we advance to the training section. 

The second thing we add to this assignment is the capability to estimate the depth of each pixel from a single image. MiDaS is the State-of-the-art "**Monocular Depth Estimation**" model that gives the grayscale depth output. 

The task of this assignment is to create a network that can perform 3 tasks simultaneously:
1. Predict the boots, PPE, hardhat, and mask if there is an image
2. Predict the depth map of the image
3. Predict the Planar Surfaces in the region

An example output:
Clockwise from top left: Raw image, Object Detection, Planes and Depth
![](https://github.com/AyusheeMittal/EVA5_MidasYoloPlanarcnn/blob/main/output.png)

For this Assignment model, We take the Encoder-Decoder type of architecture as the output we aim to get is "Dense". So we take the Resnet101 Pretrained architecture as a common Encoder for both Yolo and Midas and add two separate branches for Yolo Decoder and Midas Decoder. We have made use of "**Transfer learning**" to finetune our model with respect to our inputs by loading the weights for the Midas layer as well as that of Yolo. The model structure looks like this - https://github.com/AyusheeMittal/EVA5_MidasYoloPlanarcnn/blob/main/model.png

The Yolo layers have been taken from the Yolov3 architecture. we have added a section of layers called "Yolo Head" to combine the encoder with the Yolo decoder. The Yolo model aims at drawing the bounding box around the object to specify its location. Each bounding box has 4 descriptors, namely 
1. Center of the box (bx, by)
2. Width (bw)
3. Height (bh)
4. Value c corresponding to the class of an object ie the "Objectness"
In the end, we use the "Non-Maximum Suppression" Algorithm to produce the final decision. It helps eliminate the unnecessary anchor boxes which are very close by performing the IOU(Intersection over Union) with the one having the highest class probability among them.
The outputs produced are of size 52x52, 26x26, and 13x13 thus the detections are made at strides of 32, 16, 8 keeping in mind the images of very low resolution, medium resolution, and high resolution respectively.

Also, we were able to integrate the planar RCNN Model for predicting the mask and generating 3-D planes. The model was taken from the NVlabs Repo. This network detects the planer regions given a single RGB image and predicts 3D plane parameters together with a segmentation mask for each planar region. A Refinet model to further refine/optimize the masks of these detected planars is also added.  However, the model has been commented in this assignment code for we couldn't integrate the version with the latest one. The train functionality is also provided which we couldn't bring to the execution stage and henceforth commented too. 

## Training
We start training by loading the Yolo Ultralytics weights for the Yolo decoder layers and Midas weights for the Midas decoder layers. 

We train the Yolo branch by freezing the Resnet101 encoder layers and the Midas layers. We run the Yolo model with image size 64x64 for 50 epochs. We get the map value as 0.2 after training 64x64. Then we run the Yolo model with image size 128x128 for another 50 epochs passing the best weights. We get the map value as 0.408. Furthermore, we then run for another 50 epochs with image size 256x256 and get the map value as 0.525. The map value doesn't increase on training further.

We then start training the Midas branch by freezing the Yolo layers and the encoder layers. we train till SSIM value is close to 1 (ie 1 - loss is as low as possible) by taking the previous best weights. Note- Because of the transformations used (Yolo's), this value was always very low. 

We then run a basic inference to see the Yolo outputs, and also to see the depth images.

## Dataset
This dataset consists of around 3500 images containing four main classes:
* Hard hats
* Vest
* Mask
* Boots
Common settings include construction areas/construction workers, military personnel, traffic policemen, etc.
Not all classes are present in an image. Also, one image may have many repetitions of the same class.
For example, a group of construction workers without helmets, but with vests and boots.

The dataset is available under- https://drive.google.com/drive/u/1/folders/1nD1cdLk5y-rpmtiXH-JeU5vLvLLYVVyp
It has three folders namely Depth, Labels, and Planes. And the main Dataset Zip file - YoloV3_Dataset.zip.

### 1. Raw images
The raw images are present in the zip folder. The images were collected by crowdsourcing and do not follow any particular naming convention.
They are also of varied sizes. There are 3591 images.
These are mostly .jpg files (< 0.5% might be otherwise)

### 2. Bounding Boxes
A Yolo compatible annotation tool was used to annotate the classes within these images.
These are present under the labels folder as text files. However please note that not all raw images have a corresponding label. There are 3527 labeled text files. A few things to note:
* Each image can contain 0 or more annotated regions.
* Each annotated region is defined by four main parameters: x, y, width, height.
* For the rectangular region, (x, y) coordinates refer to the top left corner of the bounding box.
* Width and Height refer to the width and height of the bounding region. The centroid of the bounding box can be calculated from this if required.
* A label file corresponding to an image is a space-separated set of numbers. Each line corresponds to one annotated region in the image.
The first column maps to the class of the annotated region (the order of the classes is as described above). The other four numbers represent the bounding boxes (ie annotated region) and stand for the x, y, width, and height parameters explained earlier. These four numbers should be between 0 and 1.

### 3. Depth images
Planes were created using this repo:
https://github.com/NVlabs/planercnn
These are .png files, make sure to handle them accordingly since the raw images are .jpg. There are 3545 planar images. The names are the same as that of the corresponding raw images.

### Note
This dataset needs to be cleaned up further.
* There are a few (<0.5%) png files among the raw images, which need to be removed (These do not have labels ie bounding boxes, nor do they have planar images).
* There are a few (<0.5%) label files that are of invalid syntax (the x,y coordinates, or the width/height are > 1). These need to be discarded.
* Final cleaned up dataset should only include data where all these three files are present for a raw image: labels text file, depth image, and planar image.
