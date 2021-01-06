import cv2 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

#
# Semantic classes in Duckietown Simulator
#
LANE_EDGE        =  255 
SIGN_HORIZONTAL  =  204
VEHICLE          =  178
LANE             =  153
PEDESTRIAN       =  127
BUILDING         =  102
NATURE           =  76
SIGN_VERTICAL    =  51
OTHER            =  0

class Features:
    
    def __init__(self, groundTruth=[]):
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.fast = cv2.FastFeatureDetector_create(nonmaxSuppression=True) 
        self.orb = cv2.ORB_create()

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        # camera orientation wrt the Scaramuzza paper for 1-point RANSAC
        self.camera_orientation_angle=15*np.pi/180
        self.move = 0.006 # The robot meves with a speed of 20 m/s 
        self.L = 0.058 # distance camera from wheels axis
        self.camera_matrix = np.array([
            305.5718893575089, 0, 303.0797142544728,
            0, 308.8338858195428, 231.8845403702499,
            0, 0, 1,
        ]).reshape((3,3))

        self.img_cur = []
        self.sem_cur = []
        self.kp_cur  = []
        self.semdes_cur  = []
        self.des_cur = []

        self.img_prev = []
        self.sem_prev = []
        self.kp_prev  = []
        self.semdes_prev  = []
        self.des_prev = []

        self.matches=[]
        self.correspondences=[]
        
        self.theta=0
        self.rho=0

        self.theta_prev=0
        self.rho_prev=0

        self.R_cur=np.eye(3)
        self.T_cur=np.array([[0],[0],[0]])

        self.R_prev=np.eye(3)
        self.T_prev=np.array([[0],[0],[0]])

        self.fig = go.FigureWidget()
        self.fig.add_scatter()
        self.fig
        self.plot_data=[]
        self.img_cur_colored=[]

        self.point_cloud= []

        self.POINT_CLOUD=False

        self.pose_truth=np.array(groundTruth)
        self.pose_truth-=self.pose_truth[0]  # start everything from 0
        self.pose_truth=list(self.pose_truth)
        
    def sem_idx(self, value):
        if value == OTHER:
            return 0
        if value == SIGN_VERTICAL:
            return 1
        if value == NATURE:
            return -1
        if value == BUILDING:
            return 2
        if value == PEDESTRIAN:
            return -2
        if value == LANE:
            return 3
        if value == VEHICLE:
            return -3
        if value == SIGN_HORIZONTAL:
            return 4
        if value == LANE_EDGE:
            return 5
        return -4

    def __orb(self):
        self.kp_cur, self.des_cur = self.orb.detectAndCompute(self.img_cur, None)

    def _corners(self):
        # Extract Shi-Tomasi corners
        #corners = cv2.goodFeaturesToTrack(self.img_cur, maxCorners=100,qualityLevel=0.1,useHarrisDetector=True)
        # Extrac Fast
        self.kp_cur = self.fast.detect(self.img_cur, None)

    def _descriptors(self):
        # BRIEF descriptor
        self.kp_cur, self.des_cur = self.brief.compute(self.img_cur, self.kp_cur)

    def estract(self, img, sem):
        self.img_cur_colored=img
        self.img_cur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.sem_cur = cv2.cvtColor(sem, cv2.COLOR_BGR2GRAY)

        # Detect Shi-Tomasi corners and BRIEF descriptors
        # self._corners()
        # self._descriptors()
        self.__orb()

        # clean keypoints keeping only the ones with a semantic descriptor
        # with only "static" classes
        kp_cur=[]
        des_cur=[]
        for idx, kp in enumerate(self.kp_cur):
            # create a semantic descriptor only 6 are the classes considered static
            sem_desc = np.zeros(6)

            # keep the semantic from a cricle around the keypoint with a radius proportional to the 
            # octave the point was extracted from
            y = np.round(kp.pt[1]).astype(int)
            x = np.round(kp.pt[0]).astype(int)

            not_good = False
            #print(kp.octave)
            for i, j in np.array([[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]]):

                # get the index of the semantic class                
                idx_value = self.sem_idx(self.sem_cur[y+j,x+i])

                if idx_value<0:
                    not_good=True
                    break

                sem_desc[idx_value] = 1

            if not_good:
                continue

            kp_cur.append(kp)
            des_cur.append(self.des_cur[idx])
            self.semdes_cur.append(sem_desc)

        self.des_cur = np.array(des_cur)
        self.kp_cur  = np.array(kp_cur)

        return self.kp_cur, self.des_cur

    def match(self):
        if self.img_prev==[]:
            #
            # The second frame is still unknown
            #
            return []

        matches = self.bf.match(self.des_cur, self.des_prev)
        matches = sorted(matches, key = lambda x : x.distance)

        if len(matches)<50:
            self.POINT_CLOUD=False


        self.matches=[]
        for m in matches:
            if (self.semdes_cur[m.queryIdx]==self.semdes_prev[m.trainIdx]).all():
                self.matches.append(m)
                self.correspondences.append([self.kp_prev[m.trainIdx].pt, self.kp_cur[m.queryIdx].pt])


        return self.matches

    # def hamming(self,a,b):
    #     #print(f"Comparing {a} and {b}")
    #     return np.count_nonzero(np.bitwise_xor(np.int0(a),np.int0(b)))
    
    def poseFromP3P(self):
        #dist_coef = np.zeros(4)
        repr_error, self.T_cur, self.R_cur = cv2.solvePnPGeneric(
            objectPoints=self.point_cloud, 
            imagePoints=self.kp_cur, 
            cameraMatrix=self.camera_matrix, 
            distCoeffs=[],
            rvecs=self.R_cur,
            tvecs=self.T_cur, 
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE)
        return self.T_cur, self.R_cur

    def triangulate(self):
        correspondences1 = np.float32([p[0] for p in self.correspondences]).T
        correspondences2 = np.float32([p[1] for p in self.correspondences]).T
        print(len(correspondences2[0]))

        self.R_prev=np.eye(3)
        self.T_prev=np.array([[0],[0],[0]])

        R_T1 = np.concatenate((self.R_prev,self.T_prev),axis=1)
        R_T2 = np.concatenate((self.R_cur,self.T_cur),axis=1)

        projMatrix1 = self.camera_matrix.dot(R_T1)
        projMatrix2 = self.camera_matrix.dot(R_T2)

        points = cv2.triangulatePoints(projMatr1=projMatrix1, projMatr2=projMatrix2, projPoints1=correspondences1, projPoints2=correspondences2)
        print(len(points[0]))
        points/=points[3]  # bethe result is in homogeneous coordinates

        self.point_cloud=(points[:3]).T
        self.point_cloud=self.point_cloud[self.point_cloud[:,2]<10]
        self.point_cloud=self.point_cloud[self.point_cloud[:,2]>-10]

        return self.point_cloud, self.img_cur_colored[np.int0(correspondences1[1]),np.int0(correspondences1[0])]

    def eight_point(self):
        if self.POINT_CLOUD:
            return self.poseFromP3P()

        self.one_point()

        # FM_8POINT for foundamental matrix
        if len(self.kp_prev)<5 or len(self.kp_cur)<5:
            print("Not enough correspondences")
            return [],[]
        
        correspondences1 = np.float32([p[0] for p in self.correspondences])
        correspondences2 = np.float32([p[1] for p in self.correspondences])

        # cam_matrix=np.array([[1.0 ,0 ,len(self.img_cur)/2],[0 ,1.0 ,len(self.img_cur[0])/2],[0 ,0 , 1]])
        # E, mask = cv2.findEssentialMat(points1=correspondences1,points2=correspondences2,cameraMatrix1=cam_matrix,cameraMatrix2=cam_matrix,dist_coeff1=None, dist_coeff2=None, method=cv2.RANSAC,prob=0.99,threshold=1.0)
        
        E, mask = cv2.findEssentialMat(points1=correspondences1, points2=correspondences2, cameraMatrix=self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        self.correspondences=[]
        for idx,i in enumerate(mask):
            if i == 1:
                self.correspondences.append([correspondences1[idx],correspondences2[idx]])
                
        _, self.R_cur, self.T_cur, mask = cv2.recoverPose(E, points1 = correspondences1, points2 = correspondences2, cameraMatrix=self.camera_matrix, mask=mask)
        
        self.T_prev=self.T_cur
        self.R_prev=self.R_cur

        self.T_cur = self.camera_matrix.dot(self.T_cur) + self.move*2*(self.R_prev.dot(self.T_prev))
        invers_cam = np.linalg.inv(self.camera_matrix)
        self.R_cur = self.R_prev.dot(self.camera_matrix.dot(self.R_cur).dot(invers_cam))

        self.triangulate()

        self.POINT_CLOUD=True

        return self.R_cur, self.T_cur, self.pose_truth[0]


    
    def one_point(self, ):
        # discretized alpha in 180 steps
        alpha_hist = np.zeros((360,), dtype=int)

        self.prev_theta = self.theta
        
        for m in self.matches:
            du = (self.kp_cur[m.queryIdx].pt[1]-self.kp_prev[m.trainIdx].pt[1])*np.sin(self.camera_orientation_angle)
            dv = (self.kp_cur[m.queryIdx].pt[0]+self.kp_prev[m.trainIdx].pt[0])
            self.theta+=-2*np.arctan(dv/du)*180/np.pi+90
            # alpha_es = np.round(-2*np.arctan(dv/du)*180/np.pi).astype(int) +180
            # alpha_hist[alpha_es]=alpha_hist[alpha_es]+1
        self.theta/=len(self.matches)
        # a_max=0
        # for idx, a in enumerate(alpha_hist):
        #     if a > a_max:
        #         self.theta = idx-180
        #print(a_max)
        
        self.prev_rho = self.rho
        self.theta=self.theta*np.pi/180

        print("Theta from one point algo")
        print(self.theta)

        self.rho = self.theta/2


        self.T_prev=self.T_cur
        self.R_prev=self.R_cur
        offset_vec = np.array([[self.L*np.cos(self.theta) + self.move*np.cos(self.rho)-self.L],
                     [self.move*np.sin(self.rho)-self.L*np.sin(self.theta)], 
                     [0]])
        self.T_cur = offset_vec + self.move*(self.R_cur.dot(self.T_prev))
        self.R_cur = self.R_prev.dot(np.array([[np.cos(self.theta), -np.sin(self.theta), 0],   
                                            [np.sin(self.theta), np.cos(self.theta), 0],
                                            [0, 0, 1]]))

        return self.R_cur, self.T_cur, self.pose_truth[0]
    

    def calcError(self):
        if not self.pose_truth==[]:
            print(f"diff {self.pose_truth[0][0]}")
            error_x=np.linalg.norm(self.pose_truth[0][0]-self.T_cur[0])
            error_y=np.linalg.norm(self.pose_truth[0][1]-self.T_cur[1])
            error_angle=np.linalg.norm(self.pose_truth[0][2]-self.T_cur[0])
            return [error_x,error_y,error_angle]
    
    

    def next(self):
        self.img_prev    = self.img_cur.copy()
        self.kp_prev     = self.kp_cur.copy()
        self.des_prev    = self.des_cur.copy()
        self.semdes_prev = self.semdes_cur.copy()
        
        self.kp_cur  = []
        self.semdes_cur  = []
        self.des_cur = []

        if not self.pose_truth==[]:
            self.pose_truth.pop(0)