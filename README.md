# Leonardo DiCaprio Detection

In this project, we developed a deep learning model to recognize images of actor Leonardo DiCaprio (Titanic, The Wolf of Wall Street, Inception). This project consists of three main components all of which result in a live feed facial detection system for Leonardo DiCaprio.

1) Creating a Comprehensive Dataset (movie_to_images.ipynb)
Our first task was to develop a comprehensive dataset to train and test our model on. This involved processing frames from movies through opencv and MTCNN that DiCaprio starred in, as well as sampling images from the internet, to develop a comprehensive dataset of 1,131 still images of Leonardo DiCaprio. Collecting our dataset in this manner allowed us to gather images of Leonardo DiCaprio from multiple angles, with different facial expressions, and throughout his career from his first interview as a 16 year old in Paris to his current interviews. We went through a similar process with actors that are not Leonardo DiCaprio to produce 1,140 images for the negative side of our dataset.

2) Developing a Robust Deep Learning Model - fine-tuning.ipynb
Our next step was to design a deep learning model specialized in recognizing Leonardo DiCaprio’s face. To do so, our model utilizes MTCNN (Multi-task Cascaded Convolutional Networks) to detect and crop faces from our dataset of images. Then, we integrate our model on a fine-tuned version of Inception ResNet V1, adjusted using an 80/20 train validation split, to align with the specific characteristics of DiCaprio's features. This specialized training enables the model to discern details and unique features that are distinctive to Leonardo, optimizing its accuracy in face recognition.

3) Integrating Our Model In a Live Video Stream - pytorch-video-stream.ipynb
Lastly, we implemented a live video stream facial recognition system using our trained model. Specifically, our program uses either a Haar cascade classifier or MTCNN for face detection. It continuously processes frames from the video stream, detects faces using the chosen method, and classifies them as either Leonardo DiCaprio or not Leonardo DiCaprio based on the results of our model. This video stream showcases the model's ability to perform real-time facial recognition on a live input, offering a practical application of deep learning in video stream processing.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [References](#references)

## Installation/Setup

Installation follows the pytorch-video-stream notebook. First, copy face_recog_2.h5 and pytorch-video-stream.ipynb to your local drive. The first file face_recog_2.h5 holds the fine tuned InceptionNet model and pytorch-video-stream is the notebook which will allow you to run the live stream. Ensure that face_recog_2.h5 is in a path that is easily accessible. We will now work exclusively in pytorch-video-stream. Run all cells up until the "Video Stream Capture" header. This will require mounting your google drive to the notebook. Under the "Video Stream Capture" header, there is the useHaar variable. Set it to True to use the Haar Cascade Classifier and False to use the MTCNN model to capture faces from the live video stream. Observe that the Haar Cascade Classifier runs significantly quicker than the MTCNN model but is relatively less accurate. It is recommended that the Haar Cascade Classifier be used and useHaar is set to True. You can now run all cells to the bottom! The final cell is under the header "Video Stream". When you run this cell, the live stream will start and face detection will begin!

## Usage

After following the above instructions, you should see a video stream under the "Video Stream" cell. As you move, you should see a white boundary box around your face. Note that our model works best when the user's face is vertically straight (with relatively small head tilt). This can be changed by activating MTCNN instead of Haar Cascade but will result in a large dip in performance. This boundary box will also display a label of whether or not you are Leonardo DiCaprio along with the confidence of the model that you are or are not Leonardo. You can now take your favorite photo of Leonardo DiCaprio and place it in front of the camera. The model works best in good lighting but will also work relatively well with dim lighting or pictures that are rotated at a slight angle. As you move around the picture of Leonardo DiCaprio, you should see the boundary box continue to follow Leonardo. 

## Implementation Details

The implementation details of our project fall mainly into 3 categories:

1) Dataset Collection
The crux of our project is a good face detector. In order for this to work, our dataset needed to be vast and robust including images of Leonardo DiCaprio from many angles, in many lighting scenarios, and from different points in his life. In order to do this, we first collected a number of YouTube videos which contained both compilations of his acting scenes (3 minutes of Leonardo DiCaprio’s terrific acting) along with interviews of his (16-Year-Old Leonardo DiCaprio FIRST Interview). These videos provided a deep dataset of both different lighting scenarios, different quality scenarios and different faces in general. We then additionally took still images from Getty images and the internet in order to get higher quality photos along with photos where the lighting was as good as it gets. 

After compiling this dataset, we used movies_to_images to go through each video and picture and applied MTCNN to capture Leo’s face. In order to improve efficiency and get a more diverse dataset, we applied MTCNN every 8 frames in Leo’s videos. This was a result of the videos we used being captured at roughly 30 frames per second. Capturing each frame would provide incredibly similar photos but we found 8 frames to both provide efficiency and diversity in faces. This resulted in a robust dataset of Leonardo’s face moving around and turning along with high quality photoshoot pictures. 

We then performed a similar process on YouTube videos from other actors (Top 10 Performances by Jake Gyllenhaal) and photoshoots of other actors for our not Leonardo DiCaprio dataset. We once again used a frame reader, this time applying MTCNN every 120 frames for an even more diverse negative dataset. This resulted in 1,131 still images of Leonardo DiCaprio and 1,140 still images of other actors.

Model Implementation
Once we had our dataset ready, we needed to create our model. Because of its robustness and accuracy on photos, we decided to use InceptionResnetV1 as our model. We initially implemented transfer learning with InceptionResnetV1. While this approach worked decently well on our training and test set, there was a lot more room for improvement. 

In order to improve our model, we utilized fine tuning on InceptionResnetV1. Our first attempt at fine tuning was with unfreezing all layers and using an Adam optimizer with a learning rate of 0.001 but this performed far worse than we expected (roughly 40% on training and validation sets) and was no better than randomly guessing Leonardo or not. 

In order to fix this, we froze all layers except for the last fully connected linear layer which would result in our model picking up on the general facial features and then specifically tuning them towards Leonardo’s features. This change was largely successful and resulted in a jump to 92% accuracy on both our training and validation test sets while not overfitting. Our final optimization was to test out different learning rates and when we reduced our learning rate to 0.0001, we found that our training time reduced to 30 minutes and our accuracy jumped to 95%.


## References

We utilized the following resources to help us with our project:
How to Use a Webcam in Google Colab: https://www.youtube.com/watch?v=YjWh7QvVH60

Google Colab Camera Capture: https://colab.research.google.com/#snippetFileIds=%2Fv2%2Fexternal%2Fnotebooks%2Fsnippets%2Fadvanced_outputs.ipynb&snippetQuery=Camera%20Capture

Face Recognition Using Transfer Learning: https://github.com/sabarish244/Face-Recognition-using-Transfer-learning

Fine Tuning an InceptionResnetV1 Model: https://github.com/timesler/facenet-pytorch/blob/master/examples/finetune.ipynb

ChatGPT: https://chat.openai.com/
