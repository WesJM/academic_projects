# Video Recognition Technology: Enhancing Golf Swing Performance for Players of All Levels

## Introduction

Since the COVID-19 pandemic, golf has become one of the fastest-growing sports in America, with 3.3 million Americans playing golf for the first time in 2022. Golf's overall reach is estimated to be 119 million people, with approximately 41.1 million Americans playing golf either on or off-course in 2022. The golf industry's economic activity is valued at $84.1 billion as of 2016, presenting significant economic and business opportunities.

The golf swing is a rigorous, full-body motion that requires fluid coordination from head to toe to achieve optimal results. To perfect their swing, golfers need to devote substantial time to training, meticulous attention to biomechanical detail, high levels of skill, sheer physical ability, and substantial practice. Most amateur golfers turn to learning and practice methods to improve their performance, typically analyzing and adjusting their golf swing via one of two common approaches: golf instructors and technology.

This research focuses on using computer vision and deep learning techniques to streamline amateur golfer swing analysis via golf swing sequencing, extending pre-existing research by McNally, et al. Golf swing sequencing identifies the eight key frames within swing videos: Address, Toe-up, Mid-backswing, Top, Mid-downswing, Impact, Mid-follow-through, Finish. After identifying the key events/images of a golf swing, this research uses MediaPipe's Pose human pose estimation technology to extract key golfer biomechanical features at each event, facilitating more accessible opportunities for improvement. By comparing these features to professional golfers and/or previous swings, golfers can better analyze and adjust their swing to improve their performance.

## Golf Swing Sequencing

This repository contains PyTorch implementions for testing a series of deep learning models for performing golf swing sequencing.

Each model is trained on split 1 **without any data augmentation** on processed videos from the golfdb database. The processed video inputs for each model are stored in the data directory under their respective model name (e.g. the optical flow processed videos are under data/videos_160_optical_flow). The data folder also contains python scripts for converting the raw golfdb videos to their processed form (e.g. data/optical_flow.py).

The python scripts to train and evaluate each model are located within their respective model directory (e.g. the SwingNet model trained and evaluated on optical flow videos is located in optical_flow_model).

### How to run

#### Model Training
Within each model folder, there is a Jupyter Notebook titled run.ipynb. This code is meant to be executed on Google Colab. To reproduce the model results, set the notebook runtime type with the following parameters: Hardware accelerator = GPU, Runtime shape = High-RAM.

#### Golf Swing Sequencing Application

1. Follow this [tutorial](https://code.visualstudio.com/docs/python/python-tutorial) to install VSCode & Python.

2. Clone this repository or copy the swing_sequence_application folder into local file directory and set this as working directory.

3.
```
# Create a virtual environment in the .venv subdirectory
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

4. 
```
pip install requirements.txt
```

5. 
```
shiny run --reload
```

If completed successfully, you should be able to open the application running on your local machine:

https://media.github.iu.edu/user/16772/files/c258ab01-dc25-4a8b-820c-f757bc12205a

### Results

Executing the run notebooks for each model should produce the following out-of-sample Percentage of Correct Events (PCE) scores:

+ Original SwingNet Model: 71.5%
+ **Optical Flow Model: 79.6%**
+ Background Removal Model: 69.0%
+ Human Pose Model: 69.5%

The resulting best model can predict the golf swing sequencing events of new videos (along with their confidence):

<img src="/images/191_optical_flow_swing_sequence.jpg" alt="Alt text" title="Optical Flow Model Predictions on 191.mp4 (out-of-sample in split 2)">

## Biomechanic Analysis
This section contains code and a trained model used to sequence the golf swing, find the frames of interest. The program then implements Mediapipe's human pose software.

### How to run
This code is meant to be executed in a Google Colab notebook environment. You must have the "mediapipe.py" file in the same level as the "models" folder, which contains the "swingnet_2000.pth.tar" model. It must also be in the same directory as the "eval.py" and "model.py" from the sequencing section. After mounting your Google Drive, you must also install mediapipe using pip into your Colab environment using the command
```
!pip install mediapipe
```
Then you may run the program with the command line code
```
%run -i 'mediapose.py' -p 'test_video.mp4' -a 'behind' -rl 'right' -p2 'data/videos_160/1221.mp4' -a2 'behind' -rl 'right' -f 'shoulder_rotation'
```
The commands are as follows: "-p" for the first video, "-a" for the first video's camera angle (behind or open), and "-rl" for the first video's golfer's handedness. The same commands with a "2" label represent the same inputs regarding the second video. If the user only desires to analyze one video, they may ignore those commands. The last command, "-f", selects the feature the user wants to display. The features available are the following:
   + shoulder_tilt
   + hip_tilt
   + spine_angle
   + hip_translation
   + shoulder_rotation
   + hip_rotation
   + hand_height
   + hand_distance
   + rhip_angle
   + rankle_angle
   + rknee_angle
   + relbow_angle
   + rwrist_angle
   + rknee_toe_translation
   + lhip_angle
   + lankle_angle
   + lknee_angle
   + lelbow_angle
   + lwrist_angle
   + lknee_toe_translation



### Results
Our output appears as follows:

<img src="/Biomechanic_Analysis/biomechanic_results.png" alt="Alt text" title="Biomechanics results">

It shows the feature from both videos and the percent difference from video 1 to video 2. The only disclaimer needed is based on how shoulder and hip rotation was calculated. Due to not being able to reliably use the z-coordinates, we measured a quasi-rotation by either dividing the shoulder (hip) width seen by the camera by the shoulder (hip) width at the address for a behind camera angle and the address width by the frame width for an open camera angle. This means that the greater the ratio, the more rotation.

## Dependencies
* [PyTorch](https://pytorch.org/)
* Numpy
* OpenCV cv2
* Mediapipe

 ## Authors & Contributions

 - Joel Klein
   + Performed initial analysis into Golf Swing analysis and proposed project idea for team.
   + Proposed data processing, swing sequencing modeling experiments, and biomechanical analysis approach.
   + Replicate existing SwingNet model and evaluation metrics.
   + Research and select model approaches for experiments.
   + Prepare a machine learning data environment for model experiments.
   + Implement new feature engineering, video preprocessing, data preparation steps:
     + Optical Flow
     + Background Removal
     + Adding human pose estimation landmark features - need to alter data preparation to perform in batch before model train.
   + Implement a proposed new modeling architecture adding human pose estimation landmark features.
   + Execute model training for proposed experiments.
   + Update evaluation code of new models to report PCE by independent features (i.e. camera angle, slow/real-time, etc.) and compare with baseline.
   + Perform bias evaluation of models.
   + Select best performing model.
   + Deploy model in developed Shiny application to take input any set of trimmed golf swing videos and detect 8 frames for comparison.
   + Oversee and manage Github repository for golf swing sequencing code.
   + Record videos of golf swing to test model on modern video technology.
   + Co-authored all sections of project proposal.
   + Authored Introduction, Background, Data, Methodology, Progress, Updated Project Plan, and References sections of interim report.
   + Authored Abstract, Introduction, Background, Data, Methodology, Golf Swing Sequencing Results, Golf Swing Sequencing Discussion, and References sections of interim report.
   
 - Gabriel Levy
    + Researched important golf swing biomechanics.
    + Authored almost all of the code for the biomechanic calculation and analysis portion of project (Wes found the angle calculation method)
    + Implemented way for users to input preferences in a generalized manner
    + Created user friendly output for biomechanic analysis
    + Created poster for poster presentation
    + Authored biomechanical analysis methods, results, and discussion sections, and the conclusion.
    
 - Wes Martin
   + Format design and co-author project proposal. 
   + Co-author project interim report.
   + Conduct optical flow algorithms research using OpenCV.
   + Developed code script to batch process optical flow videos.
   + Performed optical flow model training and predication experiments.
   + Research calculation method of biomechanical insights using Mediapipe Pose.
   + Created script to calculate biomechanical angle analysis using identified key frames.
   + Applied suggestions to draft project poster.

## References

The baseline code edited to perform the deep learning modeling experiments in this repo is sourced from wmcnally/golfdb:
```
@InProceedings{McNally_2019_CVPR_Workshops,
author = {McNally, William and Vats, Kanav and Pinto, Tyler and Dulhanty, Chris and McPhee, John and Wong, Alexander},
title = {GolfDB: A Video Database for Golf Swing Sequencing},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2019}
}
```
```
https://github.com/Pradnya1208/Squats-angle-detection-using-OpenCV-and-mediapipe_v1
```
