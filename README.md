# Dynamic_Traffic_Zonebased
TrafficFlowAnalyzer is a Python-based tool that leverages the YOLO object detection model to analyze traffic flow in video footage. It allows users to define specific traffic zones (light, moderate, heavy) and counts the number of vehicles passing through each zone in real-time. vehicles are assigned  weight based on their effect on traffic  flow.

Key Features:
Zone Selection: Allows users to manually define traffic zones on a video frame.
Real-Time Traffic Counting: Detects and counts vehicles in different traffic zones using the YOLO model.
Weighted Traffic Analysis: Provides weighted counts of vehicles based on their type (car, bus, truck) to estimate traffic flow and congestion.
Dual Video Output: Generates two video outputs—one with bounding boxes around detected vehicles and one without, while keeping the zone markings intact.

Usage:
Load your video and define traffic zones interactively or use default zones.
Run the analysis to track and count vehicles across the defined zones.
View and save the results as videos with and without bounding boxes.

Requirements:
Python 3.x
OpenCV
Ultralytics YOLO
cvzone

Installation

Step 1 : open command prompt paste pip install opencv-python numpy ultralytics cvzone
Step 2 : In the command prompt go to location of main.py   (command - cd Dynamic_Traffic_Control_Zonebased)
Step 3 : open any editor paste the video file location in Create Zone.py
Step 4: run the program click on five points to create the zone then press 'q' 
Step 5: In main.py replace the values in line no -89 ->  light_traffic = np.array([[3,477], [634,473], [585,199], [84,249], [6,473]], np.int32)  
Step 7: similarly repeat step 4 and 5  for moderate and heavy traffic 
Step 8: run main.py press 'n' 
Step 9: for new video , run main.py press 'y' then a total of 15 clicks , will create 3 zones only for that video 
