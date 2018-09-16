# **Behavioral Cloning** 




**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


---
---
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality


#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode


My project includes the following files:
* [video.mp4](https://github.com/uv10000/P4/blob/master/video.mp4) shows the final CNN performing on Track 1 for substantially more than 1 entire lap. It follows the road quite nicely and seems to be able to perform an indefinite number of laps. 
*  [model.py](https://github.com/uv10000/P4/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/uv10000/P4/blob/master/drive.py) for driving the car in autonomous mode. Nearly unmodified, but I increased the set point for velocity to the max. 
* [model.h5](https://github.com/uv10000/P4/blob/master/model.h5) containing a trained convolution neural network 
* Your are reading [writeup_P4_behavioural_cloning_Ulrich_Voll.md](https://github.com/uv10000/P4/blob/master/writeup_P4_behavioural_cloning_Ulrich_Voll.md) summarizing the results


For reproducing my results by running model.py, please put the files provided by Udacity (Track 1) one level higher, i.e. 

"ls ../data_provided_by_udacity/" 

should yield:

 "driving_log.csv  IMG"

#### 2. Submission includes functional code
For a first preview you might want to download and play the [video.mp4](https://github.com/uv10000/P4/blob/master/video.mp4) 

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
(Provided the Unity-simulator is up and running.)


#### 3. Submission code is usable and readable

The [model.py](https://github.com/uv10000/P4/blob/master/model.py) file contains the code for training and saving the convolution neural network. 

The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network along the lines of the [NVIDIA paper](https://arxiv.org/abs/1604.07316) recommended in class (cf. model.py lines 97-165). 

My earlier attempts trying to adapt variations of the LeNet architecture were unsuccessful. 

The model includes RELU layers to introduce nonlinearity (code lines 112 and others, cf. the respective argument to the Conv2D commands, and the "model.add(Activation('relu'))" lines for the fully connected layers).

The data is normalized in the model using a Keras lambda layer (code lines 100 f). 




#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on the data set provided by Udacity.



The model contains dropout layers in order to reduce overfitting (model.py lines 109, 155,159). 
REMARK: For the final version I have set the dropout probability to 0.0 as this yields better driving performance.


The original amount of 8036 datasets  (= 8036 * 3 images, due to the additional left/right cameras) were split into a training set containing 6428 datasets and a validation set containing 1608 datasets (=20%). 
```
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2) 
```

 In order to ensure that the model was not overfitting various methods of data augmentation were employed.


 Data augmentation takes place in a python generator (model.py lines 29 - 79 ):

 ``` python 
 def generator(samples, batch_size=32):  
     ...
     yield sklearn.utils.shuffle(X_train, y_train)
 ``` 
The generator hands out batches of desired size upon reading an array of lines. 


In more detail, the generator does the following
For a first preview you might want to download and play the [video.mp4](https://github.com/uv10000/P4/blob/master/video.mp4) 

 * initially, it  performes random shuffle of all samples (line 32)
 * the input array "samples" containing lines read from the .csv file is cut into batches of size "batch_size". (line 34). These batches are treated  one after the other as follows.
 
 * Fore each batch and for each  line extract three images, corresponding to center, left and right camera, as well as the center angle. I used ndimage.imread, but I left various alternatives as commented out lines in the code. (model.py lines 40-50) 

* create "fictitious" values for left_angle and right angle from center_angle by adding an offset called "correction" (lines 51-56). I tuned the parameter to a value of 0.003. Attempts (unsuccessful) using a multiplicative correction are commented out. 

* Optionally, upon the flag "use_all_cameras", either only the center values for image and angle are appended to the batch, or all six values (center, left, right / image, steering angle). (56-63)

* Optionally, upon the flag "augment_by_flipping", all values are doubled by adding the respective mirror image of both image / angle. (64-75)

* Finally the batch consisting of x and y values including the augmented values is converted to a numpy array, randomly permuted and handed out by the yield-comman. 

 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. In fact it now seems to be able to go on forever on Track 1. However it fails badly, and almost immediately on Track 2. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 170).

I included dropout using a parameter of 0.1 (= 0.9 keep_prob, I had to learn this ...). Though I do not think this was mission critical, i.e. the model trained without dropout performed better than with dropout, on Track 1 that is. 
Remark: just before the first submission I decided to set the dropout probability to 0.0 because the car drives better this way (less wiggling). 

As suggested in class, I used the images of the left and the right cameras. Fictitious steering angles were computed using an offset parameter "correction" which I set to 0.003 in the final version (corresponding to a steering wheel (sic) angle of roughly 3°). 

In my interpretation, this seems to encourage the system to return back to the middle, similar to a P-Controller somehow molded into the end-to-end system. An alternative method serving the same end would have been to deliberately drive "off centre" in self generated training runs, and then "showing" the system how to steer back into the middle, by recording only the stretches of time where the car is being brought back to centre and ommiting the stretches of time where it is deliberately moved away from the centre of the lane. 

 One might ask if it were better to separate these two issues by letting the CNN merely provide a desired heading, and a dedicated "low-level"-controller do the actual steering. But this would mean abondoning the "end-to-end" approach. I would be grateful to you reviewers if you could comment on what you think is better. 

As suggested in class, I included the mirror image for every image in the training set, created an appropriately reflected steering angles accordingly. 

Both employing additional cameras and reflections are happening in the generator as described above.

Finally I applied clipping as suggested. However this did not have a big effect. Possibly the training time went down (smaller modell) but performance without clipping was even better, according to my impression. Clearly inference can be performed faster in the presence of clipping in that less convolutions have to be performed. I think it does not greatly reduce model size though, as only the parameters of the convolultions have to be stored in a CNN. 


#### 4. Appropriate training data

 I used the training data provided by udacity, setting 20% aside as a validation set as described above. 

 I also augmented the training data as described above.


---
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to try out architectures of increasing complexities. 

Following your instructions in class the very first attempt was a single Layer NN in order to see that this kind of system was able to perform some vaguely reasonably steering at all. 

My second step was to use a convolution neural network model similar to the LeNet model used previously. It seems plausible with starting as simple a network as possible. Also this was suggested in class. 

The LeNet model was fitting quite well, no overfitting, and driving quite nicely and smoothely on Track 1. However, the vehicle turned right into the sand just after the bridge, every single time,  and I could not improve on this by merely tuning parameters. 

I was not sure if I should focus on getting more training data (I was having troubles with acquiring data on my own, as I did not have a joystick and am finding it very hard to acquire reasonable data via keyboard) or on introducing a more complicated network architecture.

I decided to start with implementing the NVIDA architecture (as suggested in class, quoted above). Fortunately, this almost immediately did the trick. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. As far as Track 1 is concerned it seems to be able to keep this up indefinitely. 

However, as mentioned before, it is a total and utter fail on Track 2. Due to the problems mentioned above (and time constraints!!!) I did not train the model on any Track 2 data. 

One may question if the model has merely learned to somehow learn Track 1 by heart, as opposed to "really learning how to drive". 

#### 2. Final Model Architecture

The final model architecture (model.py lines 81-165) consisted of a convolution neural network with the following layers and layer sizes:



| Layer         		    |     Description	 | Shape        					            | line number in model.py
|:---------------------:|:--------------------------------------:|:------------------:|:---------- |
| Input         		   |      | 160x320x3 RGB image   							      |     94-100                 |
| Normalization     	| lambda x: x/127.5 - 1. | 160x320x3 	  |            100          |
| cropping (optional)     	| 50 rows from above, 20 from below | 90x320x3 	  |      104-107                |
| dropout	(optional)				        | 0.1 probability |	90x320x3											                        |       109            |
| Convolution 5x5     	| 2x2 stride, valid padding, RELU    |43x158x24	  |                112 ff      |
| Convolution 5x5     	| 2x2 stride, valid padding, RELU    |20x77x36	  |                120 ff      |
| Convolution 5x5     	| 2x2 stride, valid padding, RELU    |8x37x48	  |                128 ff      |
| Convolution 3x3     	| 1x1 stride, valid padding, RELU    |6x35x64	  |                135 ff      |
| Convolution 3x3     	| 1x1 stride, valid padding, RELU    |4x33x64	  |                143 ff      |
| Flatten        		    | 4x33x64 = |  8448        									  |              151        |
| Fully connected		    | incl. RELU + dropout (optional 0.1) | 100        									          |          153 ff            |
| Fully connected		    | incl. RELU + dropout (optional 0.1) | 50        									          |          157 ff            |
| Fully connected		    | incl. RELU   | 10        									          |          161 ff            |
| Fully connected		    |   steering angle       (a regression problem!)    | 1        									          |          165 ff            |

This is essentially the architecture provided by the aforementioned NVIDIA research paper.

Note that there are no pooling layers. (I would like to understand why they seem to be dispensable.)

I added some dropout layers but I do not think they were mission critical. There was no overfitting (on Track 1!), even with dropout probability set to 0.0. Remark: Just before submission I decided to disable dropout as the results were better without!

#### 3. Creation of the Training Set & Training Process

As mentioned above I merely used the NVIDIA training data plus data augmentation as suggested.

Some 8000 lines in the data set correspont to some 24000 images plus corresponding angles,  when including the right and left camera values. By reflecting the images/angles I used a total of some 50000 images. 

I randomly shuffled the (original!) data set and put 20% of the data into a validation set. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.

 <p align="center">
  <img width="400" src="./training_validation_errors.png">
</p>


This plot shows that both validation and training error are small from the very start (on the scale of epochs at least). Presumably convergenc happens so fast that it does not reflect on such a coarse time scale. Indeed during the first steps of the optimization one can see values for training and validation error that are substatially higher, decaying rapidly.



------------------
---------------------------
## Critical Discussion and Loose Threads


#### 1. More training data / Track 2
Clearly it would be desirable to accquire more training data, in particular for Track 2. However due to my time constraints I do not see a way of how to achieve this. 

-> It would be nice if you could provide data for Track 2 similarly as you did for Track 1 - for those of us with tight time constraintes. 

-------------

#### 2. End-To-End vs more classical approach, specifically low-level-control

As mentioned before it may be that the model merely "memorized" Track 1, rather than "properly learned" how to steer in a general environment. 

I would like to know your (the revievers) opinion(s) on wether the end-to-end approach is a true alternative to a more standard pipeline, like:
* fusion
* perception
* localization and mapping
* path planning
* low level control

As alluded to above, the end-to-end-system effectively performs (among other tasks) the task of a low level controller, implicitly. Is it plausible to assume that a CNN can learn to do better than a controller designed by the principles of state of the art control theory? And even if it works well in many cases, how can one prove correctnes and robustness in all situations that may practically occur? Admittedly, a huge chunk of hand-written code man also behave in an unexpected manner in some situations, even if very good testing-mechanisms have been put in place. Validation and verification is a touchy business in either case. 

--------
#### 3. Validation error suspiciously (?) low

I am puzzled about the fact that the validation error is low from the very start. Possibly it converges so fast that it closely approaches the training error within the first epoch. (But even setting the epoch size to lower values seems to confirm the picture that both errors are close to each other for the very start)


----
#### 4. Regularisation, where to apply dropout
Remark: Just before the first submission I decided to turn off dropout because it yields better results this way.
Concerning dropout: Apparently it makes no difference for my model on Track 1 if I turn off dropout. I think it drives even more steadily without! Using the side cameras definitely helped with LeNet. I have not attempted to turn of the side cameras in the presence of the NVIDIA-net-architecture. 
Moreover I have no clear concept of where to apply dropout layers. 

I did not have the time to try L2 regularisation. Due to time constraints. And then there was no pressing need for regularisation as mentioned above. 

----
#### 5. Clipping/Crooping did not help for performance 
Clipping sounded like a good idea but it (slightly, subjectively) deteriorated driving performance. Admittedly clipping has a sustantial impact on model size, or at least on inference time. Possible making the images coarser would have also helped (=somewhat equivalent to pooling layers?! ). The advice to use the coarsest level of resolution when recording training data also seems to point in this direction.


-----
#### 6. Regularisation, where to apply dropout

Why did I get away without using any pooling layers? I did not use any as there were no in the NVIDIA paper. But I do not have an understanding of when and why I should apply them. May be you can explain!


-----
#### 7. No transformation to other colour space
Other than in the NVIDA paper I did not apply any colour-space conversions. I used RGB right away, successfully.  



-----
#### 8. No correlation between consecutive timesteps
Each picture is treated separately, and in total isolation.  But clearly pictures (and even more so the vehicle state) will evolve continuously.   
Clearly this information is "thrown away" in the present algorithm. What would be a good way of making use of this. RNN? Averaging over the last few images/angles? Or even making predictions like in a Kalman filter based on a vehicle model?

