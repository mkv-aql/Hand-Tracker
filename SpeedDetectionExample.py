import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Calculate the initial centroid of the object
centroid = np.mean(prev_pts, axis=0)
prev_centroid = centroid

# Get the starting time
t1 = cv2.getTickCount()

while True:
    ret, frame2 = cap.read()
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None, winSize=(15, 15), maxLevel=2)

    # Select good points
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    # Find the centroid of the moving object
    centroid = np.mean(good_new, axis=0)

    # Calculate the distance between the current and previous centroids
    distance = np.linalg.norm(centroid - prev_centroid)

    # Calculate the time elapsed between the current and previous frames
    t2 = cv2.getTickCount()
    time_elapsed = (t2 - t1) / cv2.getTickFrequency()

    # Calculate the speed of the moving object
    speed = distance / time_elapsed

    # Draw a circle at the centroid of the moving object
    cv2.circle(frame2, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)

    # Print the speed of the moving object
    cv2.putText(frame2, f"Speed: {speed:.2f} pixels per second", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the video frame
    cv2.imshow('frame', frame2)

    # Update the previous frame and centroids
    prev_gray = next_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)
    prev_centroid = centroid
    t1 = cv2.getTickCount()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
