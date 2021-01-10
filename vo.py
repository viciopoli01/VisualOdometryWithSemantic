#
# Opencv version 4.4.0
#
import cv2
from VisualOdometry import VisualOdometry
import numpy as np 

from class_plot import Plot
import threading



import json
import random
from gevent import sleep
import csv


POSES = "/home/viciopoli/Desktop/VisionAlgorithm/VO050121/poses.csv"
CAMERA_MATRIX = np.array([
            305.5718893575089, 0, 303.0797142544728,
            0, 308.8338858195428, 231.8845403702499,
            0, 0, 1,
        ]).reshape((3,3))

print("Starting Visual Odometry...")

text = "http://localhost:5000"
target = "http://localhost:5000"
print(f"Open the following link to visualize the data: \u001b]8;;{target}\u001b\\{text}\u001b]8;;\u001b\\")

def ws_handler(ws):
    sleep(2)  # delay between data collection calls
    
    step = 1
    groud_truth_pose = []

    # read the ground truth
    with open(POSES) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            groud_truth_pose.append(row)
    

    # features = VisualOdometry(groud_truth_pose[:],step)
    features = VisualOdometry(intrinsic=CAMERA_MATRIX, groundTruth=groud_truth_pose, step=step)


    img1 = cv2.imread(f'dt_dataset/45.png', cv2.IMREAD_GRAYSCALE)
    sem1 = cv2.imread(f'dt_dataset/seg_45.png', cv2.IMREAD_UNCHANGED)
    
    k1 = features.estract(img1, sem1)
    print(len(k1))
    
    features.next()


    for idx,i in enumerate(range(120,500,step)):
        print(f"step n. {i}")
        sleep(1)

        print(f'dt_dataset/'+str(i)+'.png')

        img_cur = cv2.imread(f'dt_dataset/'+str(i)+'.png',cv2.IMREAD_GRAYSCALE)
        sem_cur = cv2.imread(f'dt_dataset/seg_'+str(i)+'.png', cv2.IMREAD_UNCHANGED)

        # Estact the features
        _, reinit_frame = features.estract(img_cur, sem_cur)

        if not reinit_frame:
            # Match the features
            k_cur,k_prev=features.match()

            # Triangulate with 5-point or with EPNP using the Point Cloud
            T, real_pose , points = features.triangulate()

            # draw the resulting pose
            plot.add_traj_point([T[0][0],0,T[2][0]],real_pose)

            # Plot the point cloud
            plot.add_point_cloud(points)
        
        # plot the image with the optical flow
        plot.draw_image(cv2.cvtColor(img_cur, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # update the graph
        data = idx
        ws.send(json.dumps(idx))

        features.next()

    sleep(2)
    exit(0)

starting_image=cv2.cvtColor(cv2.imread(f'dt_dataset/45.png',cv2.IMREAD_GRAYSCALE), cv2.COLOR_BGR2RGB)

plot = Plot(ws_handler,starting_image)
plot.start_server()


