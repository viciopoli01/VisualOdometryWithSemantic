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

        Logger().printInfo("    ==== Starting VO pipeline ====    ")

        # create BFMatcher object
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.orb = cv2.ORB_create()

        # Duckiebot camera orientation
        self.camera_orientation_angle=15*np.pi/180


        self.camera2world = np.array([ 1.0000000,  0.0000000,  0.0000000,
                                    0.0000000, -0.2588190, -0.9659258,
                                    0.0000000,  0.9659258, -0.2588190]).reshape((3,3))
        
        self.camera_matrix = intrinsic.copy()
        # features variables for the current frame
        self.img_cur = []
        self.kp_cur  = []
        self.des_cur  = []
        self.semdes_cur  = []

        # features variables for the previous frame
        self.img_prev = []
        self.kp_prev  = []
        self.des_prev  = []
        self.semdes_prev  = []

        self.correspondences=[]

        # Init the starting Rotation and translation.
        self.R_cur=self.camera2world #np.eye(3)
        self.T_cur=np.zeros((3,1))

        self.R_prev=[]
        self.T_prev=[]

        # Point cloud variables
        self.point_cloud = []
        self.point_cloud_kp = []
        self.point_cloud_sem_descriptor = []
        self.point_cloud_correspondences = []
        # self.point_cloud_R = np.eye(3)
        # self.point_cloud_T = np.array([[0],[0],[0]])
    
        # Flags for the point cloud update
        self.POINT_CLOUD=False
        self.POINT_CLOUD_ITER=0

        # Get the ground truth and start everything from zero.
        self.pose_truth=np.array(groundTruth)
        self.pose_truth-=self.pose_truth[0]  # start everything from 0
        self.pose_truth=list(groundTruth)
        # how many frames are skipped
        self.step=step

        # iteration counter
        self.iter_count=0

        # flag if the tracked features are few
        self.STARTING_FRAME=True
        self.ORB_MATCH=True
        self.MIN_FEATURES=1500
        self.FirstPointCloud=True

        self.Framskip=0
        self.KLTwinSize=(20,20)

        self.total_scale=0

    def scale(self):
        #
        #   Calculating the scale for the translation vector.
        #

        dx=self.pose_truth[(self.step)*self.iter_count+1][0]-self.pose_truth[(self.step)*self.iter_count][0]
        dy=self.pose_truth[(self.step)*self.iter_count+1][1]-self.pose_truth[(self.step)*self.iter_count][1]
        dtheta=self.pose_truth[(self.step)*self.iter_count+1][2]-self.pose_truth[(self.step)*self.iter_count][2]

        return np.sqrt(dx*dx+dy*dy), dtheta

    def sem_idx(self, value):
        if value == OTHER:
            return 0
        if value == SIGN_VERTICAL:
            return 1
        if value == NATURE:
            return 2
        if value == BUILDING:
            return 3
        if value == PEDESTRIAN:
            return 4
        if value == LANE:
            return 5
        if value == VEHICLE:
            return 6
        if value == SIGN_HORIZONTAL:
            return 7
        if value == LANE_EDGE:
            return 8
        return 0

    def __skipFrame(self):
        if self.Framskip>0:
            self.Framskip-=1
        self.total_scale+=self.scale()[0]
        return not self.Framskip==0

    def __corners(self):
        """ Computes FAST corners and ORB descriptors """
        kp_cur = self.orb.detect(self.img_cur,None)
        kp_cur, self.des_cur = self.orb.compute(self.img_cur, kp_cur)
        kp_cur = np.array([x.pt for x in kp_cur], dtype=np.float32)
        self.refineConers(kp_cur)
        Logger().printInfo("    New Keypoints found.    ")

    def __LKT(self):
        """ Tracks the keypoints using Lucas-Kanade-Tomasi """

        assert(len(self.semdes_prev)==len(self.kp_prev)), "Semantic descriptor and keypoints' vector must have the same lenght"
        
        # track the FAST corners using Lucas-Kanade method
        kp_cur, status, err = cv2.calcOpticalFlowPyrLK(self.img_prev, self.img_cur, self.kp_prev, None, winSize  = self.KLTwinSize, maxLevel = 3 , criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
        
        status = status.reshape(status.shape[0])
        # use only the good points and semantic descriptor
        self.refineConers(kp_cur)
        self.kp_cur = kp_cur[status == 1]
        
        self.kp_prev = self.kp_prev[status == 1].copy()
        self.semdes_prev = np.array(self.semdes_prev)[status == 1].copy()

        pc_=[]
        if self.POINT_CLOUD:
            for idx,i in enumerate(status):
                if i == 1:
                    pc_.append(self.point_cloud[idx])
            self.point_cloud=pc_
        
        Logger().printInfo("    Features tracked and point cloud updated.   ")

    def __getSemanticDescriptor(self):
        """ Create the binary semantic descriptors """
        self.semdes_cur=[]

        for idx, kp in enumerate(self.kp_cur):
            # create a semantic descriptor only 6 are the classes considered static
            sem_desc = np.zeros(9)

            y = np.round(kp[1]).astype(int)
            x = np.round(kp[0]).astype(int)

            #
            # the circle around the keypoint should be proportional to the octave of the keypoint descriptor,
            # anyway since here there is no descriptor we use a pre-defined pattern
            #
            for i, j in np.array([[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]]):

                # get the index of the semantic class
                # 
                if (y+j)<480 and (x+i)<640:              
                    idx_value = self.sem_idx(self.sem_cur[y+j][x+i])
                    sem_desc[idx_value] = 1

            self.semdes_cur.append(sem_desc.copy())

    def estract(self, img, sem):
        """
        Estract feature points.
        img = current frame
        [sem = semantic image if available]
        """

        self.img_cur = img.copy()
        if sem==[]:
            sem = np.zeros_like(img)
        else:
            self.sem_cur = cv2.cvtColor(sem, cv2.COLOR_BGR2GRAY).copy()

            
        # Redetect features or track them
        if self.STARTING_FRAME: # the first frame needs new features
            # Detect Shi-Tomasi corners
            Logger().printWarning("    Starting Frame    ")
            self.firstFrameInit()
            self.STARTING_FRAME=False
            return self.kp_cur, True

        elif self.ORB_MATCH:
            if self.__skipFrame():
                Logger().printWarning("     Skipping frames...     ")
                return [], True
            Logger().printInfo("Ready to match ORB keypoint for Point Cloud Initialization")
            self.ORB_MATCH=False
            self.__corners()
            self.__getSemanticDescriptor()
            return self.kp_cur, False
        else:
            Logger().printWarning("     LKT tracking    ")
            self.__LKT()
            self.__getSemanticDescriptor()
            return self.kp_cur, False

    def match(self):
        """
        Match descriptors for ORB and remove tracked features with different semantic descriptor.
        """
        if self.POINT_CLOUD==True:
            self.__matchCloud()
        else :
            self.__matchNoCloud()

        return self.kp_cur, self.kp_prev

    def __matchNoCloud(self):
        """ Match ORB descriptors """
        kp_cur  = []
        kp_prev = []
        sem_cur = []
        sem_prev = []

        matches = self.bf.match(self.des_cur, self.des_prev)

        for m in matches:
            if (self.semdes_cur[m.queryIdx]==self.semdes_prev[m.trainIdx]).all(): # and self.semdes_prev[m.trainIdx][4]==0 and self.semdes_prev[m.trainIdx][6]==0:
                kp_cur.append(self.kp_cur[m.queryIdx])
                kp_prev.append(self.kp_prev[m.trainIdx])
                sem_cur.append(self.semdes_cur[m.queryIdx])
                sem_prev.append(self.semdes_prev[m.trainIdx])  
        
        self.kp_cur=np.array(kp_cur.copy(), dtype=np.float32)
        self.kp_prev=np.array(kp_prev.copy(), dtype=np.float32)
        self.semdes_cur=np.array(sem_cur.copy())
        self.semdes_prev=np.array(sem_prev.copy())        

        assert(len(self.semdes_prev)==len(self.semdes_cur)), "Problem in semantic matching"

    def __matchCloud(self):
        """ Match semantic descriptors """

        kp_cur  = []
        kp_prev = []
        sem_cur = []
        sem_prev = []
        pc_=[]

        assert(len(self.semdes_prev)==len(self.semdes_cur)), "Problem in semantic matching"

        self.correspondences=[]
        for i in range(len(self.semdes_cur)):
            if (self.semdes_cur[i]==self.semdes_prev[i]).all():
                kp_cur.append(self.kp_cur[i])
                kp_prev.append(self.kp_prev[i])
                sem_cur.append(self.semdes_cur[i])
                sem_prev.append(self.semdes_prev[i])
                pc_.append(self.point_cloud[i])
                self.correspondences.append([[self.point_cloud[i][0],self.point_cloud[i][1],self.point_cloud[i][2]], 
                                            [self.kp_cur[i][0],self.kp_cur[i][1]]])

        self.point_cloud=np.array(pc_).copy()
        self.kp_cur=np.array(kp_cur, dtype=np.float32)
        self.kp_prev=np.array(kp_prev, dtype=np.float32)
        self.semdes_cur=np.array(sem_cur)
        self.semdes_prev=np.array(sem_prev)

    def poseFromEPnP(self):
        self.POINT_CLOUD_ITER+=1
        #dist_coef = np.zeros(4)
        imgPoints=np.float32([m[1] for m in self.correspondences])

        objPoints=np.float32([m[0] for m in self.correspondences])

        Logger().printInfo("We got {0} correspondences.".format(len(self.correspondences)))

        imgpts, jac = cv2.projectPoints(objPoints, self.rotation_matrix_to_attitude_angles(self.R_cur), self.T_cur, self.camera_matrix, None)
        
        # for p in imgpts:
        #     print("ORCA")
        #     image = cv2.circle(self.img_cur, (p[0][0],p[0][1]), radius=5, color=(255, 255, 255), thickness=-1)

        # cv2.imshow("sasa",image)
        # cv2.waitKey(0)

        # repr_error, r_vec, t_vec, inl = cv2.solvePnPRansac(
        #     objectPoints = objPoints,
        #     imagePoints = imgPoints,
        #     cameraMatrix = self.camera_matrix,
        #     distCoeffs = None,
        #     reprojectionError=0.01,
        #     confidence=0.999,
        #     useExtrinsicGuess=True,
        #     rvec=self.rotation_matrix_to_attitude_angles(self.R_cur),
        #     tvec=self.T_cur,
        #     flags=cv2.SOLVEPNP_ITERATIVE)

        repr_error, r_vec, t_vec = cv2.solvePnP(
            objectPoints = objPoints,
            imagePoints = imgPoints,
            cameraMatrix = self.camera_matrix,
            distCoeffs = None,
            useExtrinsicGuess=True,
            rvec=self.rotation_matrix_to_attitude_angles(self.R_cur),
            tvec=self.T_cur,
            flags=cv2.SOLVEPNP_ITERATIVE)


        Logger().printSuccess("     Translation vector from point cloud:  {0}   ".format(np.linalg.norm(t_vec)))

        # imgpts, jac = cv2.projectPoints(objPoints, r_vec, t_vec, self.camera_matrix, None)
        
        # for p in imgpts:
        #     print("ORCA")
        #     image = cv2.circle(self.img_cur, (p[0][0],p[0][1]), radius=5, color=(255, 255, 255), thickness=-1)

        # cv2.imshow("sasa",image)
        # cv2.waitKey(0)

        # kp_cur=[]
        # pc=[]
        # semdes_cur=[]
        # for i in inl:
        #     print(i)
        #     kp_cur.append(self.kp_cur[i[0]])
        #     pc.append(self.point_cloud[i[0]])
        #     semdes_cur.append(self.semdes_cur[i[0]])

        # self.kp_cur=np.array(kp_cur, dtype=np.float32).copy()
        # self.semdes_cur=np.array(semdes_cur).copy()
        # self.point_cloud=np.array(pc).copy()


        # Logger().printInfo("    Inliers    {0}".format(inl))
        scale  = self.scale()[0]
        Logger().printInfo("    PnP went ok    {0}".format(repr_error))
        #print(repr_error)
        
        if repr_error:
            self.T_cur = self.point_cloud_T + self.point_cloud_R.dot(t_vec.copy())
            self.R_cur = self.point_cloud_R.dot(cv2.Rodrigues(r_vec)[0].copy())

        self.point_cloud_sem_descriptor=self.semdes_cur.copy()
        self.point_cloud_kp=self.kp_cur.copy()

        #self.R_cur = self.R_cur
        
        Logger().printInfo("    USING EPnP    ")
        # Logger().printInfo("    R    {0}".format(self.R_cur))
        # Logger().printInfo("    T    {0}".format(self.T_cur))
        Logger().printInfo("    using the point cloud    ")

        return self.R_cur, self.T_cur

    def pointCloudStatus(self):
        if len(self.correspondences)<15 or self.POINT_CLOUD_ITER==20:
            Logger().printWarning(" Point cloud needs to be reinitialized ")
            self.POINT_CLOUD=False
            self.FirstPointCloud=False
            self.POINT_CLOUD_ITER=0
            return True
        return False

    def triangulate(self):
        #  
        # Save current pose
        #
        self.T_prev=self.T_cur.copy()
        self.R_prev=self.R_cur.copy()
        
        if self.POINT_CLOUD:
            
            if self.pointCloudStatus():
                self.firstFrameInit()
            else:
                self.poseFromEPnP()
            
            return self.T_cur, self.pose_truth[self.step-1], self.point_cloud


        Logger().printInfo("Running 5-Point RANSAC algorithm")
        if len(self.kp_prev)<=5 or len(self.kp_cur)<=5:
            Logger().printFail("Not enough inliers detected....")
            self.firstFrameInit()
            gt = self.pose_truth[self.iter_count]
            exit(0)
            return  self.T_cur, [gt[0],gt[1],0], self.point_cloud 
        


        kp_prev, kp_cur = self.normalizePoints()
        #
        # Estimate the Essential Matrix
        #findEssentialMat
        E, mask = cv2.findEssentialMat(points1=kp_prev, points2=kp_cur, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=0.001)
        
        #
        # Recover pose uses decomposeEssentialMat and performs the Cheirality check (the features must be in front of the camera) on the possible Rotation maticess
        #
        # Logger().printInfo("Essential matrix: {0}".format(E))
        if np.count_nonzero(mask)<=5:
            Logger().printFail("Not enough inliers detected....")
            self.firstFrameInit()
            gt = self.pose_truth[self.iter_count]
            exit(0)
            return  self.T_cur, [gt[0],gt[1],0], self.point_cloud 

        rep_err, R_cur, T_cur, mask = cv2.recoverPose(E, points1 = kp_prev, points2 = kp_cur, mask = mask)
        Logger().printFail("Reprojection error from recover Pose : {0}".format(rep_err))

        # self.R_cur=self.camera_matrix.dot(R_cur).dot(np.linalg.inv(self.camera_matrix))
        # self.T_cur=self.camera_matrix.dot(T_cur)


        Logger().printFail("scale is {0}".format(self.total_scale))

        self.R_cur = self.R_prev.dot(R_cur).copy()
        self.T_cur = (self.T_prev + self.total_scale*self.R_prev.dot(T_cur)).copy()

        self.total_scale=0 # reset total scale

        kp_prev = []
        kp_cur  = []
        sem_cur = []
        sem_prev= []
        for i,v in enumerate(mask):
            if v == 1:
                kp_prev.append(self.kp_prev[i])
                kp_cur.append(self.kp_cur[i])
                sem_cur.append(self.semdes_cur[i])
                sem_prev.append(self.semdes_prev[i])
        
        self.kp_prev = np.array(kp_prev.copy())
        self.kp_cur  = np.array(kp_cur .copy())
        self.semdes_cur =  np.array(sem_cur .copy())
        self.semdes_prev = np.array(sem_prev.copy())

        p_matr1 = self.camera_matrix.dot(np.hstack((self.R_prev,self.T_prev)))
        # p_matr1 = self.camera_matrix.dot(np.hstack((np.eye(3),np.zeros((3,1)))))


        p_matr2 = self.camera_matrix.dot(np.hstack((self.R_cur,self.T_cur)))

        point_cloud = cv2.triangulatePoints(projMatr1=p_matr1, projMatr2=p_matr2, projPoints1=self.kp_prev.T, projPoints2=self.kp_cur.T)
        
        # p_matr1_inv = self.camera_matrix.dot(np.hstack((self.R_prev.T,-self.T_prev)))
        # point_cloud=p_matr1_inv.dot(point_cloud)

        point_cloud/=point_cloud[3]  # the result is in homogeneous coordinates
        point_cloud=np.hstack((self.R_prev, self.T_prev)).dot(point_cloud)
        self.point_cloud=(point_cloud[:3].T).copy()

        Logger().printInfo("Point cloud initialized")
        # print("Printing cloud")
        # print(self.point_cloud)
        self.POINT_CLOUD=True
        self.point_cloud_sem_descriptor=self.semdes_cur.copy()
        self.point_cloud_kp=self.kp_cur.copy()
        # Logger().printInfo("Mask cloud? :\n{0} \n{1}".format(mask,len(mask)))
        # Logger().printInfo("Point cloud :\n{0} \n{1}".format(self.point_cloud,len(self.point_cloud)))

        Logger().printInfo("Point cloud created.")

        self.point_cloud_T=self.T_prev.copy()
        self.point_cloud_R=self.R_prev.copy()

        # print(scale)
        # print(self.T_cur)
        # print(self.R_cur)
        # if self.kp_cur.shape[0]<self.MIN_FEATURES:
        #     self.__corners()

        gt = self.pose_truth[self.iter_count]

        return  self.T_cur, [gt[0],gt[1],0], self.point_cloud

    def refineConers(self,corners):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.kp_cur = cv2.cornerSubPix(self.img_cur, corners, (11,11),(-1,-1),criteria)


    def normalizePoints(self):
        kp_cur = cv2.undistortPoints(np.expand_dims(self.kp_cur, axis=1), cameraMatrix=self.camera_matrix, distCoeffs=None)
        kp_prev = cv2.undistortPoints(np.expand_dims(self.kp_prev, axis=1), cameraMatrix=self.camera_matrix, distCoeffs=None)
        return kp_cur, kp_prev

    def calcError(self):
        if not self.pose_truth==[]:
            # print(f"diff {self.pose_truth[self.step-1][0]}")
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

    def firstFrameInit(self):
        self.ORB_MATCH=True
        self.__corners()
        self.__getSemanticDescriptor()
        self.__reset()
        self.Framskip=3

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
        self.des_prev    = self.des_cur

        self.kp_cur  = []
        self.semdes_cur  = []
        self.img_cur=[]