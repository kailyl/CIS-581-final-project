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

After following the above instructions, you should see a video stream under the "Video Stream" cell. As you move, you should see a white boundary box 

## Implementation Details

Guidelines for contributing to the project and how to submit pull requests.

## References

Information about the project's license and any usage restrictions.


