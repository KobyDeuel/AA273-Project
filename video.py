import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

#set up maker
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
dp = cv2.aruco.DetectorParameters()
dp.minMarkerPerimeterRate = 0.001
dp.maxMarkerPerimeterRate = 10.0
detector = cv2.aruco.ArucoDetector(aruco_dict, dp)

#load video
cap = cv2.VideoCapture("robot_run.mp4")
if not cap.isOpened():
    raise FileNotFoundError(f"failed to open file")

#get fps and frames
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

rows = []
current_frame = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #get marker from detector
    corners, ids, _ = detector.detectMarkers(frame)
    timestamp = current_frame / fps

    #loop to get coords and heading from maker position
    if ids is not None:
        for i, coords in enumerate(corners):
            if ids[i][0] != 0:
                continue
            pts = coords[0]
            x = float(np.mean(pts[:, 0]))
            y = float(np.mean(pts[:, 1]))
            direction = (pts[0] + pts[1]) / 2 - np.array([x, y])
            heading = float(np.degrees(np.arctan2(direction[1], direction[0]))) -90
            rows.append([timestamp, x, y, heading])

    current_frame += 1
    if current_frame % 100 == 0:
        print(f"{current_frame}/{total_frames}")

cap.release()

with open("tracking.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "x_px", "y_px", "heading_deg"])
    for r in rows:
        w.writerow([f"{r[0]:.3f}", f"{r[1]:.1f}", f"{r[2]:.1f}", f"{r[3]:.1f}"])
        
print("Saved")
