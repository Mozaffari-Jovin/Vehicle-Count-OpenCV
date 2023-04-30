#### Importing relevant packages
# cmd: pip install numpy opencv-python ultralytics
import cv2
import numpy as np
from ultralytics import YOLO
import time
t0 = time.time()

#### YOLO model YOLOv8n.pt/YOLOv8s.pt/YOLOv8m.pt/YOLOv8l.pt/YOLOv8x.pt
model = YOLO("models/YOLOv8s.pt")

#### Reading the frames from the video file
cap = cv2.VideoCapture('TrafficCamera.mp4') 

# Crossing window position
up_line_position = 250
down_line_position = up_line_position + 60
down_route_position = 550

# objTracker function computes the distance (Euclidean distance) between two center points of an object in 
# the current frame and the previous frame, and if the distance is smaller than the threshold distance, 
# it certifies that the object in the previous frame is the same object in the present frame.
def objTracker(num1, num2, num3):
	
	if len(num2) == 0: # when num2 becomes empty since no object or objects other than motorcycle, car, and truck are detected 
		num2 = [[0, 0]]
	
	point1 = np.array(num1)

	dists = []
	for point in num2:
		# print(point)
		point2 = np.array(point)
		dist = np.linalg.norm(point1 - point2) # Euclidean distance 
		dists.append(dist)

	if min(dists) < num3: # num3 is the threshold distance
		return True # object is the same
	else:
		return False 

motorcycle_n = 0
car_n = 0

obj_centers_pre_frame = [[0, 0]]

while True:
	ret, frame = cap.read() 
	# ret is a boolean variable that returns true if the frame is available
	# frame is an image array vector captured based on the default frames per second defined explicitly or implicitly
	if ret:
		frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
		ih, iw, channels = frame.shape

		## draw the crossing window
		cv2.rectangle(frame, (0, up_line_position), (down_route_position, down_line_position), (255, 0, 255), 1)
		frame_observation = frame[up_line_position:down_line_position, 0:down_route_position, :]
		
        ## Object detection
		results = model.predict(source=frame_observation) 
		# print(type(results)) # <class 'list'>
		# print(results[0].boxes.xyxy, results[0].boxes.cls) # box position and ID of the detected object
		# print(len(results[0].boxes.cls))

		obj_centers = []
		# print(obj_centers)
		Threshold = 20 # threshold distance
		for result in results:
			# print(type(result)) # <class 'ultralytics.yolo.engine.results.Results'>
			for (obj_xyxy, obj_cls) in zip(result.boxes.xyxy, result.boxes.cls): # two concurrent loops
				# print(obj_cls, type(obj_cls)) # <class 'torch.Tensor'>
				obj_cls = int(obj_cls)# tensor to integer

				if obj_cls == 3: # 3 and 0 are the motorcycle and person's ID in YOLO
					x1 = obj_xyxy[0].item()
					y1 = obj_xyxy[1].item()
					x2 = obj_xyxy[2].item()
					y2 = obj_xyxy[3].item()
				    # print(x1, y1, x2, y2, type(x1)) # <class 'float'>
					
					obj_center = [(x1 + x2) / 2, (y1 + y2) / 2] # center point of object
					# print(obj_center)
					obj_centers.append(obj_center)
					
					cv2.rectangle(frame_observation, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
					
					# print(obj_centers_pre_frame)
					same_object_detected = objTracker(obj_center, obj_centers_pre_frame, Threshold)
					# print(same_object_detected)
					if not(same_object_detected):
						motorcycle_n += 1

				elif (obj_cls == 2) or (obj_cls == 7): # 2 and 7 are the car and truck's ID in YOLO
					x1 = obj_xyxy[0].item()
					y1 = obj_xyxy[1].item()
					x2 = obj_xyxy[2].item()
					y2 = obj_xyxy[3].item()

					obj_center = [(x1 + x2) / 2, (y1 + y2) / 2] 
					obj_centers.append(obj_center)

					cv2.rectangle(frame_observation, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
					
					# print(obj_centers_pre_frame)
					same_object_detected = objTracker(obj_center, obj_centers_pre_frame, Threshold)
					# print(same_object_detected)
					if not(same_object_detected):
						car_n += 1

		obj_centers_pre_frame = obj_centers.copy() # obj_centers in the previous center
		# print(obj_centers_pre_frame)
		# print(car_n)

		# show the time and text
		t1 = time.time() - t0
		t1_str = str(round(t1, 2))
		cv2.putText(frame, f"Cars: {car_n}  Motorcycles: {motorcycle_n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		cv2.putText(frame, f"Time: {t1_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		
		# show the frames
		cv2.imshow('Traffic management', frame)
		q = cv2.waitKey(1) # wait for 1 second to return a frame
		if q == ord("q"): # press q key on the webcam screen to quit 
			break
cap.release() # close cap
cv2.destroyAllWindows() # this function allows users to destroy or close all windows at any time after exiting the script.