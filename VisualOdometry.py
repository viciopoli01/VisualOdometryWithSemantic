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

class Logger:
    #
    # This class is an helper for logging. Different methods are available.
    #

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def printWarning(self, text):
        text = str(text)
        print(self.WARNING +text+self.ENDC)
    
    def printSuccess(self, text):
        text = str(text)
        print(self.OKGREEN +text+self.ENDC)
    
    def printFail(self, text):
        text = str(text)
        print(self.FAIL +text+self.ENDC)

    def printInfo(self, text):
        text = str(text)
        print(self.OKBLUE +text+self.ENDC)


class VisualOdometry:
    
    def __init__(self, intrinsic, groundTruth=[], step=1):

        self.orb = cv2.ORB()

        # Duckiebot camera orientation
        self.camera_orientation_angle=15*np.pi/180
        
        self.camera_matrix = intrinsic
        # features variables for the current frame
        self.img_cur = []
        self.kp_cur  = []
        self.semdes_cur  = []
        # features variables for the previous frame
        self.img_prev = []
        self.kp_prev  = []
        self.semdes_prev  = []
        self.correspondences=[]
        
        self.theta=0
        self.rho=0

        self.theta_prev=0
        self.rho_prev=0

        # Init the starting Rotation and translation.
        self.R_cur=np.eye(3)
        self.T_cur=np.array([[0],[0],[0]])

        self.R_prev=[]
        self.T_prev=[]

        # Point cloud variables
        self.point_cloud = []
        self.point_cloud_kp = []
        self.point_cloud_sem_descriptor = []
        self.point_cloud_correspondences = []
        self.point_cloud_R = np.eye(3)
        self.point_cloud_T = np.array([[0],[0],[0]])
    
        # Flags for the point cloud update
        self.POINT_CLOUD=False
        self.POINT_CLOUD_REFRESH=10

        # Get the ground truth and start everything from zero.
        self.pose_truth=np.array(groundTruth)
        self.pose_truth-=self.pose_truth[0]  # start everything from 0
        self.pose_truth=list(groundTruth)
        # how many frames are skipped
        self.step=step

        # iteration counter
        self.iter_count=0

        # flag if the tracked features are few
        self.LOOSE_THE_TRACK=True
        self.MIN_FEATURES=1500
        self.FirstPointCloud=True



        self.Framskip=0
        self.KLTwinSize=(25, 25)

    def scale(self):
        #
        #   Calculating the scale for the translation vector.
        #

        dx=self.pose_truth[(self.step-1)*self.iter_count][0]-self.pose_truth[(self.step)*self.iter_count][0]
        dy=self.pose_truth[(self.step-1)*self.iter_count][1]-self.pose_truth[(self.step)*self.iter_count][1]
        dtheta=self.pose_truth[(self.step-1)*self.iter_count][2]-self.pose_truth[(self.step)*self.iter_count][2]

        return np.sqrt(dx*dx+dy*dy), dtheta

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

    def __skipFrame(self):
        if self.Framskip>0:
            self.Framskip-=1
        return not self.Framskip==0

    def __corners(self):
        # Extract Shi-Tomasi corners
        self.kp_prev=[]
        self.semdes_prev=[]
        self.kp_cur = cv2.goodFeaturesToTrack(
            self.img_cur,
            maxCorners = 500,
            qualityLevel = 0.3,
            minDistance = 7,
            blockSize = 7,
            useHarrisDetector=False)
        self.kp_cur = np.array([x[0] for x in self.kp_cur], dtype=np.float32)
        self.LOOSE_THE_TRACK = False


        

    def __LKT(self):
        assert(len(self.semdes_prev)==len(self.kp_prev)), "Semantic descriptor and Points vector must have the same lenght"

        
        # track the Shi-Tomasi corners using Lucas-Kanade method
        self.kp_cur, status, err = cv2.calcOpticalFlowPyrLK(self.img_prev, self.img_cur, self.kp_prev, None, winSize  = self.KLTwinSize, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        status = status.reshape(status.shape[0])
        # use only the good points and semantic descriptor
        self.kp_cur = self.kp_cur[status == 1]
        
        self.kp_prev = self.kp_prev[status == 1]
        self.semdes_prev = np.array(self.semdes_prev)[status == 1]

        pc_=[]
        if self.POINT_CLOUD:
            for idx,i in enumerate(status):
                if i == 1:
                    pc_.append(self.point_cloud[idx])
            self.point_cloud=pc_


    def __getSemanticDescriptor(self):
        # clean keypoints keeping only the ones with a semantic descriptor with only "static" classes
        kp_cur  = []

        # remove the wrong correspondences with the previous keypoint and semantic desc
        kp_prev = []
        sem_prev = []
        self.semdes_cur=[]
        for idx, kp in enumerate(self.kp_cur):
            # create a semantic descriptor only 6 are the classes considered static
            sem_desc = np.zeros(6)

            y = np.round(kp[1]).astype(int)
            x = np.round(kp[0]).astype(int)

            # if the feature has a "non-static" class
            not_good = False
            #
            # the circle around the keypoint should be proportional to the octave of the keypoint descriptor,
            # anyway since here there is no descriptor we use a pre-defined pattern
            #
            for i, j in np.array([[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]]):

                # get the index of the semantic class
                # 
                if (y+j)<480 and (x+i)<640:              
                    idx_value = self.sem_idx(self.sem_cur[y+j,x+i])

                    if idx_value<0:
                        not_good=True
                        break

                    sem_desc[idx_value] = 1

            if not_good:
                continue
                
            if not self.kp_prev==[]:
                kp_prev.append(self.kp_prev[idx])
                sem_prev.append(self.semdes_prev[idx])

            kp_cur.append(kp)
            self.semdes_cur.append(sem_desc)


        self.kp_prev=np.array(kp_prev)
        self.semdes_prev=np.array(sem_prev)
        self.kp_cur  = np.array(kp_cur)



    def estract(self, img, sem):
        """
        Estract feature points.
        img = current frame
        [sem = semantic image if available]
        """
        self.img_cur = img
        if sem==[]:
            sem = np.zeros_like(img)
        else:
            self.sem_cur = cv2.cvtColor(sem, cv2.COLOR_BGR2GRAY)

            
        # Redetect features or track them
        if self.LOOSE_THE_TRACK: # the first frame needs new features
            # Detect Shi-Tomasi corners
            Logger().printWarning("Reinitializing features to track")
            self.__corners()
            self.__reset()
            self.Framskip=10
            self.KLTwinSize=(50, 50)
        else:
            if self.__skipFrame():
                Logger().printWarning("Skipping frames")
                return [], True
            Logger().printWarning("Tracking features")
            self.__LKT()
            KLTwinSize=(25, 25)

        self.__getSemanticDescriptor()

        return self.kp_cur, not self.kp_prev.size>0 # flag to see if we can match or not.

    def match(self):
        """
        Remove tracked features with different semantic descriptor.
        """
        if self.POINT_CLOUD==True:
            self.__matchCloud()
        else :
            self.__matchNoCloud()

        return self.kp_cur, self.kp_prev

    def __matchNoCloud(self):
        kp_cur  = []
        kp_prev = []
        sem_cur = []
        sem_prev = []
        assert(len(self.semdes_prev)==len(self.semdes_cur)), "Problem in semantic matching"

        # mask = self.semdes_cur==self.semdes_prev

        for i in range(len(self.semdes_cur)):
            if (self.semdes_cur[i]==self.semdes_prev[i]).all():
                kp_cur.append(self.kp_cur[i])
                kp_prev.append(self.kp_prev[i])
                sem_cur.append(self.semdes_cur[i])
                sem_prev.append(self.semdes_prev[i])

        self.kp_cur=np.array(kp_cur, dtype=np.float32)
        self.kp_prev=np.array(kp_prev, dtype=np.float32)
        self.semdes_cur=np.array(sem_cur)
        self.semdes_prev=np.array(sem_prev)

    def __matchCloud(self):
        kp_cur  = []
        kp_prev = []
        sem_cur = []
        sem_prev = []
        assert(len(self.semdes_prev)==len(self.semdes_cur)), "Problem in semantic matching"

        # mask = self.semdes_cur==self.semdes_prev
        self.correspondences=[]
        for i in range(len(self.semdes_cur)):
            if (self.semdes_cur[i]==self.semdes_prev[i]).all():
                kp_cur.append(self.kp_cur[i])
                kp_prev.append(self.kp_prev[i])
                sem_cur.append(self.semdes_cur[i])
                sem_prev.append(self.semdes_prev[i])
                self.correspondences.append([[self.point_cloud[i][0],self.point_cloud[i][1],self.point_cloud[i][2]], 
                                            [self.kp_cur[i][0],self.kp_cur[i][1]]])

        self.kp_cur=np.array(kp_cur, dtype=np.float32)
        self.kp_prev=np.array(kp_prev, dtype=np.float32)
        self.semdes_cur=np.array(sem_cur)
        self.semdes_prev=np.array(sem_prev)

    def poseFromEPnP(self):
        #dist_coef = np.zeros(4)
        imgPoints=np.float32(np.ascontiguousarray([m[1] for m in self.correspondences]).reshape((len(self.correspondences),1,2)))

        objPoints=np.float32([m[0] for m in self.correspondences])
        
        T_cur=np.float32(self.T_cur)
        R_cur=np.float32(self.R_cur)

        # Logger().printFail("T cur : {0}".format(self.T_cur))

        repr_error, r_vec, t_vec, _ = cv2.solvePnPGeneric(
            objectPoints=objPoints,
            imagePoints=imgPoints,
            cameraMatrix=self.camera_matrix, 
            distCoeffs=None,
            rvec=self.rotation_matrix_to_attitude_angles(self.R_cur),
            tvec=self.T_cur, 
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_EPNP)
            

        scale  = self.scale()[0]
        if self.FirstPointCloud:
            self.T_cur  = scale*t_vec[0]
            self.R_cur,_= cv2.Rodrigues(r_vec[0])
        else:
            self.T_cur = self.T_prev + self.point_cloud_R.dot(self.T_cur)
            self.R_cur = self.R_cur.dot(self.point_cloud_R)


        self.point_cloud_sem_descriptor=self.semdes_cur.copy()
        self.point_cloud_kp=self.kp_cur.copy()

        #self.R_cur = self.R_cur
        
        Logger().printInfo("    USING EPnP    ")
        # Logger().printInfo("    R    {0}".format(self.R_cur))
        # Logger().printInfo("    T    {0}".format(self.T_cur))
        Logger().printInfo("    using the point cloud    ")

        return self.R_cur, self.T_cur

    def pointCloudStatus(self):
        if len(self.correspondences)<20 or self.iter_count%self.POINT_CLOUD_REFRESH==0:
            Logger().printWarning(" Point cloud needs to be reinitialized ")
            self.POINT_CLOUD=False
            self.LOOSE_THE_TRACK=True
            self.FirstPointCloud=False
            return True
        return False

    def triangulate(self):
        #
        # Save current pose
        #
        self.T_prev=self.T_cur
        self.R_prev=self.R_cur
        
        if self.POINT_CLOUD:
            self.poseFromEPnP()
            if self.pointCloudStatus():
                self.__corners()
                self.__getSemanticDescriptor()
                self.__reset()
                self.Framskip=10
            return self.T_cur, self.pose_truth[self.step-1], self.point_cloud


        Logger().printInfo("Running 5-Point RANSAC algorithm")
        if len(self.kp_prev)<5 or len(self.kp_cur)<5:
            Logger().printFail("Not enough inliers detected....")
            self.LOOSE_THE_TRACK = True
            gt = self.pose_truth[self.iter_count]

            return  self.T_cur, [gt[0],gt[1],0], self.point_cloud 
        
        #
        # Estimate the Essential Matrix
        #
        E, mask = cv2.findEssentialMat(points1=self.kp_cur, points2=self.kp_prev, cameraMatrix=self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        #
        # Recover pose uses decomposeEssentialMat and performs the Cheirality check (the features must be in front of the camera) on the possible Rotation maticess
        #
        # Logger().printInfo("Essential matrix: {0}".format(E))
        if np.count_nonzero(mask)<5:
            Logger().printFail("Not enough inliers detected....")
            self.LOOSE_THE_TRACK = True
            gt = self.pose_truth[self.iter_count]

            return  self.T_cur, [gt[0],gt[1],0], self.point_cloud 

        rep_err, self.R_cur, self.T_cur, mask = cv2.recoverPose(E, points1 = self.kp_cur, points2 = self.kp_prev, cameraMatrix=self.camera_matrix, mask = mask)
        
        p_matr1 = self.camera_matrix.dot(np.hstack((self.R_prev,self.T_prev)))
        p_matr2 = self.camera_matrix.dot(np.hstack((self.R_cur,self.T_cur)))
        point_cloud = cv2.triangulatePoints(projMatr1=p_matr1, projMatr2=p_matr2, projPoints1=self.kp_cur.T, projPoints2=self.kp_prev.T)
        
        p_matr1_inv = self.camera_matrix.dot(np.hstack((self.R_prev.T,-self.T_prev)))
        #point_cloud=p_matr1.dot(point_cloud)
        point_cloud/=point_cloud[3]  # the result is in homogeneous coordinates

        self.point_cloud=(point_cloud[:3]).T
        Logger().printInfo("PointCloud Found")
        # print("Printing cloud")
        # print(self.point_cloud)
        self.POINT_CLOUD=True
        self.point_cloud_sem_descriptor=self.semdes_cur.copy()
        self.point_cloud_kp=self.kp_cur.copy()
        # Logger().printInfo("Mask cloud? :\n{0} \n{1}".format(mask,len(mask)))
        # Logger().printInfo("Point cloud :\n{0} \n{1}".format(self.point_cloud,len(self.point_cloud)))

        Logger().printInfo("Point cloud created.")

         
        scale  = self.scale()[0]
        Logger().printFail("scale is {0}".format(scale))
        self.T_cur = self.T_prev + scale*self.R_prev.dot(self.T_cur)
        self.R_cur = self.R_cur.dot(self.R_prev)

        self.point_cloud_T=self.point_cloud_T+self.point_cloud_R.dot(self.T_prev)
        self.point_cloud_R=self.R_prev.dot(self.point_cloud_R)

        # print(scale)
        # print(self.T_cur)
        # print(self.R_cur)
        # if self.kp_cur.shape[0]<self.MIN_FEATURES:
        #     self.__corners()
    


        gt = self.pose_truth[self.iter_count]

        return  self.T_cur, [gt[0],gt[1],0], self.point_cloud

    def calcError(self):
        if not self.pose_truth==[]:
            print(f"diff {self.pose_truth[self.step-1][0]}")
            error_x=np.linalg.norm(self.pose_truth[self.step-1][0]-self.T_cur[0])
            error_y=np.linalg.norm(self.pose_truth[self.step-1][1]-self.T_cur[1])
            error_angle=np.linalg.norm(self.pose_truth[self.step-1][2]-self.T_cur[0])
            return [error_x,error_y,error_angle]
    
    def rotation_matrix_to_attitude_angles(self, R):
        import math
        cos_beta = math.sqrt(R[2,1] * R[2,1] + R[2,2] * R[2,2])
        validity = cos_beta < 1e-6
        if not validity:
            alpha = math.atan2(R[1,0], R[0,0])    # yaw   [z]
            beta  = math.atan2(-R[2,0], cos_beta) # pitch [y]
            gamma = math.atan2(R[2,1], R[2,2])    # roll  [x]
        else:
            alpha = math.atan2(R[1,0], R[0,0])    # yaw   [z]
            beta  = math.atan2(-R[2,0], cos_beta) # pitch [y]
            gamma = 0                             # roll  [x]  
        return np.array([alpha, beta, gamma])*180/np.pi   

    def next(self):
        self.iter_count+=1
        if self.Framskip==0: 
            self.__reset()

        if not self.pose_truth==[]:
            for _ in range(self.step):
                self.pose_truth.pop(0)

    def __reset(self):
        self.img_prev    = self.img_cur.copy()
        self.kp_prev     = self.point_cloud_kp if self.POINT_CLOUD else self.kp_cur.copy()
        self.semdes_prev = self.semdes_cur.copy()

        self.kp_cur  = []
        self.semdes_cur  = []
        self.img_cur=[]