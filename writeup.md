#**Behavioral Cloning**

##Writeup Pedro Marques

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.jpg "Model Visualization"
[image6]: ./images/original_image.png "Normal Image"
[image7]: ./images/flipped_image.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 5x5 filters and depths between 24 and 48 (model.py lines 99-101) and 2 3x3 filters with 64 depth

The model includes RELU layers to introduce nonlinearity (code lines 99-122), and the data is normalized in the model using a Keras lambda layer (code line 87).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 112 and 116).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 128). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 128).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used mainly center lane driving, as my model was failing to perform the hard left turns to the left on the second track, I tried capturing multiple snapshots of vehicle doing a hard left turn, and also on every other turn where it had difficulties.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to have create a neural network that could drive a vehicle by cloning driving style.

My first step was to use a convolution neural network model similar to the LeNet Architecture. I thought this model might be appropriate because I used it on the previous project for predicting the traffic signs classes and it worked quite well in doing so. But for this project I used the NVIDIA architecture as part of a recommendation from the course material and the performance was really great!

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that it would use dropout before the first two fully-connected layers

Then I added a cropping layer to slice the input images so that scenery that would distract the feature extraction process were removed, mainly the trees, the sky and the hood of the car.

The final step was to run the simulator to see how well the car was driving around track one. On the first attempt, before adding dropout, or cropping, the car would just go straight out of the road, turning left and going crazy. With the addition of those layers the driving behavior improved substantially.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 83-130) consisted of a convolution neural network with the following layers and layer sizes:
  3 5x5 Convolutional Layers each with 2x2 stride and followed by a RELU activation layer
  2 3x3 Convolutional Layers each followed by a RELU activation layer
  1 Flatten Layer
  4 Dropout layers the first 3 followed each by a RELU activation layer, and the first two, each preceded by a dropout layer, number of neurons for each layer (1st - 100, 2nd - 50, 3rd - 10 and last is the output)

Here is a visualization of the architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6]

I have not performed recovery laps for this model, I wanted first to see how it would behave without them, and thankfully it did well.

Then I repeated this process on track two in order to get more data points, but I only recorded one lap on it.

To augment the data set, I also flipped images and angles thinking that this would help the model generalize better and to not be prone to turn to the direction you have more turns, for example, the first track is mostly turns to the left, so the model could become inclined to only turn left. Here is an image that has then been flipped:

![alt text][image6]
Original
![alt text][image7]
Flipped

After the collection process, I had X number of data points. I then preprocessed this data by dividing each of the data points by 127.5 and subtracting 1 to it.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the difference between the validation loss and training loss, it would not increase after the second epoch, it would oscillate up and down. I used an adam optimizer so that manually training the learning rate wasn't necessary.


### Experiments
I have created the model also using the generator, It was easier to train a lot more data, but ultimately the results were poor. With fairly less data the model I created without the generator performed better than the other one, I have uploaded the file "model_with_generator.py" in case you want to take a look and see if I did something wrong. 
