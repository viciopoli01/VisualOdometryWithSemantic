#
# Opencv version 4.4.0
#
import cv2
from VisualOdometry import Features
import numpy as np 

from class_plot import Plot
import threading




# def extract_features(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     corners = cv2.goodFeaturesToTrack(gray, 30, 0.01, 10)
#     return np.int0(corners)

# def descriptors(img, corners):
#     hog = cv2.HOGDescriptor(
#         _winSize=cv2.Size(16,16),
#         _blockSize=cv2.Size(8,8),
#         _blockStride=cv2.Size(8,8),
#         _cellSize=cv2.Size(8,8)
#         )
#     descs = hog.compute(img,locations=corners)
#     print(len(corners))
#     print(len(descs))
#     print(descs)

# def check_match(des1, des2):
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1,des2)
#     matches = sorted(matches, key = lambda x:x.distance)
#     return matches

# def LKT(img1, img2, corners1, corners2):
    
#     p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, corners1,None , maxLevel = 2)
    
#     mask = np.zeros_like(img1)
#     good_new = p1[st==1]
#     good_old = corners[st==1]
#     # draw the tracks
#     for i,(new,old) in enumerate(zip(good_new,good_old)):
#         a,b = new.ravel()
#         c,d = old.ravel()
#         mask = cv2.line(mask, (a,b),(c,d), (0,255,0), 2)
#         frame = cv2.circle(frame,(a,b),5,(0,255,0),-1)
#     img = cv2.add(frame,mask)
#     cv2.imshow('frame',img)


# CAMERA_TILT=15*np.pi/180
# R_cv=np.array([[0, -1, 0],
#             [-np.sin(CAMERA_TILT), 0, -np.cos(CAMERA_TILT)],
#             [ np.cos(CAMERA_TILT), 0, -np.sin(CAMERA_TILT)]])

# print(R_cv)

# R_cv_inv=np.linalg.inv(R_cv)
# print(R_cv_inv)

# print(np.dot(R_cv_inv, R_cv))

# exit(0)

        

# x = threading.Thread(target=prova(), args=(1,))
# x.start()



import json
import random
from gevent import sleep
import csv
def ws_handler(ws):
    sleep(2)  # delay between data collection calls
    # data = random.uniform(0, 1)  # collect data (simulated by random number)
    # ws.send(json.dumps(data))  # send data
    
    groud_truth_pose = []
    with open("train_1500/pose.csv") as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            groud_truth_pose.append(row)
    features = Features(groud_truth_pose[200:])

    point_cloud=[]

    traj=[]

    img1 = cv2.imread(f'train_1500/samples/200.png')
    sem1 = cv2.imread(f'train_1500/samples/seg_200.png', cv2.IMREAD_UNCHANGED)

    for idx,i in enumerate(range(205,350,5)):
        sleep(0.1)
        img2 = cv2.imread(f'train_1500/samples/{i}.png')
        sem2 = cv2.imread(f'train_1500/samples/seg_{i}.png', cv2.IMREAD_UNCHANGED)

        kp1,_=features.estract(img1, sem1)

        features.next()

        kp2,_= features.estract(img2, sem2)

        matches=features.match()

        img3 = cv2.drawMatches(img2,kp2,img1,kp1, matches, None,flags=2)
        cv2.imshow("res3",img3)

        # R, T, real_pose = features.one_point()
        R, T, real_pose = features.eight_point()
        plot.add_traj_point([T[0][0],T[1][0],T[2][0]],real_pose)

        err = features.calcError()
        plot.add_err_point(err)

        points, color = features.triangulate()
        plot.add_point_cloud(points)
        #plot.draw_point(points, color)
        #print(f"R : {R}, T: {T}")

        for m in matches:
            c = (0,256,3)
            thickness=2
            r=1
            end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
            end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int))
            cv2.arrowedLine(img1, end2, end1, c, thickness)
            cv2.circle(img1, end1, r, c, thickness)
            cv2.circle(img1, end2, r, c, thickness)

        cv2.imshow("res1",img1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        img1 = img2.copy()
        sem1 = sem2.copy()

        data = idx  # collect data (simulated by random number)
        ws.send(json.dumps(idx))
    
    exit(0)


plot = Plot(ws_handler)
plot.start_server()

plot.draw_line()

plot.show()
cv2.waitKey(0)
# cv2.solvePnPRansac()
