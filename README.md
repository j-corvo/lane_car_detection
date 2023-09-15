# Autonomous Driving - Lane and Car Detection
## Description

The aim of this project was to develop an application, that from a video and making use of the GPU for image processing, detects lanes and vehicles. Furthermore, it marks the car depending on the lane they are detected: 
1) Draws a red square, in case the other vehicles are in the same lane.
2) Draws a green square, in case they are detected in other lanes (see the picture).

![image](https://github.com/j-corvo/Lane_Car_Detection/assets/52609366/068cc6dc-9ceb-47d8-95af-7bcc9a365b49) 

## How it works

In case you want to explore and improve the code, the application works as the following:

1) Install pycharm (recommended) or any other IDE that can run python code.
2) Install the required packages through the `pip3 install` command. You should only need to install the OpenCv library and OpenCl library:
   - `pip3 install opencv-python`
   - `pip3 install pyopencl`
3) Run the code using the play button of the pycharm IDE (in case you are using pycharm).

The video should start playing and you should see:

1) Two lines (blue and red) detecting the lanes.
2) A square (red and or green) identifying the cars.  

