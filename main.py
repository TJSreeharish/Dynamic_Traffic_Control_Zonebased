import cv2
import numpy as np
from ultralytics import YOLO
import math
import cvzone


# Global variables
polygon_points = []
selected_zones = []

# Load the YOLO model
model = YOLO(r'C:\Users\tjsre\OneDrive\Desktop\ty\yolov10n.pt')

# Classes for YOLO
classes = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
    5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
    10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
    20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
    25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
    30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 
    34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 
    65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 
    69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 
    78: 'hair drier', 79: 'toothbrush'
}

# Function to handle mouse clicks
def mouse_callback(event, x, y, flags, param):
    global polygon_points, selected_zones
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(polygon_points) < 15:
            polygon_points.append((x, y))
            print(f"Point Added: (X: {x}, Y: {y})")
        if len(polygon_points) == 5:
            selected_zones.append(np.array(polygon_points, np.int32))
            polygon_points = []
        elif len(polygon_points) == 10:
            selected_zones.append(np.array(polygon_points[5:], np.int32))
            polygon_points = polygon_points[:5]
        elif len(polygon_points) == 15:
            selected_zones.append(np.array(polygon_points[10:], np.int32))

# Function to select zones
def select_zones(video_path):
    global polygon_points, selected_zones
    polygon_points = []
    selected_zones = []
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('Select Zones', frame)
    cv2.setMouseCallback('Select Zones', mouse_callback)

    while True:
        if len(selected_zones) == 3:
            cv2.destroyWindow('Select Zones')
            cap.release()
            return selected_zones

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyWindow('Select Zones')
    cap.release()
    return []

# Define the video path
video_path = r"C:\Users\tjsre\OneDrive\Desktop\ty\hackathon\test4.mp4"

# Ask the user if they want to select zones
change_zones = input("Any changes to the zones? (y/n): ").lower()
if change_zones == 'y':
    zones = select_zones(video_path)
    if len(zones) < 3:
        print("Error: Not all zones were selected. Please try again.")
        exit()
    light_traffic, moderate_traffic, heavy_traffic = zones
else:
    light_traffic = np.array([[3,477], [634,473], [585,199], [84,249], [6,473]], np.int32)
    moderate_traffic = np.array([[101,232], [619,190], [576,132], [164,136], [101,232]], np.int32)
    heavy_traffic = np.array([[162,136], [553,137], [451,45], [229,50], [164,135]], np.int32)

# Create a capture object for processing
cap = cv2.VideoCapture(video_path)

# Define video writers for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_with_boxes = cv2.VideoWriter('output_with_boxes.mp4', fourcc, 20.0, (640, 480))
output_without_boxes = cv2.VideoWriter('output_without_boxes.mp4', fourcc, 20.0, (640, 480))

# Main loop
# Define vehicle weights
vehicle_weights = {
    'car': 1,
    'bus': 2,
    'truck': 3
}

# Initialize weighted vehicle counts
light_traffic_weighted = 0
moderate_traffic_weighted = 0
heavy_traffic_weighted = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("video not found")
        break

    frame = cv2.resize(frame, (640, 480))
    frame_no_boxes = frame.copy()
    results = model(frame)

    # Reset vehicle counts
    light_traffic_count = 0
    moderate_traffic_count = 0
    heavy_traffic_count = 0

    light_traffic_weighted = 0
    moderate_traffic_weighted = 0
    heavy_traffic_weighted = 0
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            centerx, centery = x1 + w // 2, y1 + h // 2 - 40
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            conf = math.ceil(confidence * 100)
            vehicle = classes[class_detect]

            if vehicle in ['car', 'bus', 'truck'] and conf >= 5:
                # Draw bounding box and label
                color = (0, 255, 0) if vehicle == 'car' else (255, 255, 0) if vehicle == 'bus' else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'{vehicle} {conf}%'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Count the vehicle in the appropriate zone with weight
                weight = vehicle_weights.get(vehicle, 1)
                
                if cv2.pointPolygonTest(light_traffic, (centerx, centery), False) >= 0:
                    light_traffic_count += 1
                    light_traffic_weighted += weight
                if cv2.pointPolygonTest(moderate_traffic, (centerx, centery), False) >= 0:
                    moderate_traffic_count += 1
                    moderate_traffic_weighted += weight
                if cv2.pointPolygonTest(heavy_traffic, (centerx, centery), False) >= 0:
                    heavy_traffic_count += 1
                    heavy_traffic_weighted += weight

    # Calculate the time estimate based on weighted counts
    time_estimate = (light_traffic_weighted * 1.2 + 
                     moderate_traffic_weighted * 1.5 + 
                     heavy_traffic_weighted * 1.8)
    cv2.polylines(frame, [light_traffic], True, (0, 255, 0), 2)
    cv2.polylines(frame, [moderate_traffic], True, (0, 255, 255), 2)
    cv2.polylines(frame, [heavy_traffic], True, (0, 0, 255), 2)
    # Display the time estimate
    cvzone.putTextRect(frame, f'Light Traffic Zone = {light_traffic_weighted}', [0, 20], thickness=1, scale=1, border=0, colorR=(0,0, 0), colorT=(0, 255, 0))
    cvzone.putTextRect(frame, f'Moderate Traffic Zone = {moderate_traffic_weighted}', [0, 60], thickness=1, scale=1, border=0, colorR=(0, 0, 0), colorT=(0, 255, 255))
    cvzone.putTextRect(frame, f'Heavy Traffic Zone = {heavy_traffic_weighted}', [0, 100], thickness=1, scale=1, border=0, colorR=(0, 0, 0), colorT=(0, 0, 255))
    cvzone.putTextRect(frame, f'time = {time_estimate:.2f} seconds', [0, 140], thickness=1, scale=1, border=1, colorR=(0, 0, 0), colorT=(0, 0, 255))
    cvzone.putTextRect(frame_no_boxes, f'Estimated time = {time_estimate:.2f} seconds', [0, 25], thickness=2, scale=2, border=1, colorR=(0, 0, 0), colorT=(0, 128, 0))
    # Draw the zones on both frames
    

    cv2.polylines(frame_no_boxes, [light_traffic], True, (0, 255, 0), 2)
    cv2.polylines(frame_no_boxes, [moderate_traffic], True, (0, 255, 255), 2)
    cv2.polylines(frame_no_boxes, [heavy_traffic], True, (0, 0, 255), 2)

    # Show the frames (with and without bounding boxes)
    cv2.imshow("Frame with Boxes", frame)
    cv2.imshow("Frame without Boxes", frame_no_boxes)

    # Write the frames to the respective video files
    output_with_boxes.write(frame)
    output_without_boxes.write(frame_no_boxes)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the resources
cap.release()
output_with_boxes.release()
output_without_boxes.release()
cv2.destroyAllWindows()
