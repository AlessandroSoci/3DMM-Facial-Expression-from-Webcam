# 3DMM-Facial-Expression-from-Webcam

##Run
- download the [repository](https://github.com/AlessandroSoci/3DMM-Facial-Expression-from-Webcam.git) (clone or zip download)
- run `main.py` or `expression_to_neutral.py`

The script will output the same window but with different application dependently if you run `main.py` or `expression_to_neutral`.

## Introduction
The main project is the deploy of an application that show an expressive face, having a neutral face as a base. 

Furthermore the application can show a neutral face, having an expressive face as a base, but it is only a experiment,
and we did not spend a lot of time with it.

## Goals
0. Recognize the face of the subject
1. Build the 3D face model
2. Apply the texture to the 3D model
3. Modify the expression

## Technologies
- The interface is developed with the PyQt Framework using Python.
- OpenCV allow to have control on camera.
- The face landmark are calculated thanks to [Python packages](https://github.com/ageitgey/face_recognition).
- 3D Morphable Model

The process for calculating the landmark face has a significant computational cost and also the building of 3D model.
This involve in a lag and in a slowdown of the application.

## Interface and Description
To provide a visual feedback to the user, the system implements a simple interface.

The user wiil see a live view of the scene on the left size of the window, and on the right part only a simple imagine.
On the top there is a Toolbox with witch the user can interact. It has the following widget:


###Working in Progess...
