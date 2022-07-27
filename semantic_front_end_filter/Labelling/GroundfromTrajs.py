from cProfile import label
from cmath import nan
from turtle import color
from cv2 import mean
import msgpack
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import numpy
from pandas import value_counts
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from numpy import asarray as ar, ndarray
from torch import true_divide, zero_
import os

# Class GFT provides utility functions to work with ground map extracted from feet trajctories. 
# The class can be instantiated by feet trajectory file (saved by extractFeetTrajsFromRosbag.py in pyenv)
# or by file saved by the calss itself. The main method is getHeight(self, x, y, method = 'sparse'), 
# given the target point, it will return the height of that point.

def moveArray(array, movex, movey):
    # a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        array = array.copy()
        array = np.roll(array, (movex, movey), axis = (0, 1))
        if(movex >= 0):
            array[:movex, :] = 0
        else:
            array[movex:, :] = 0
        if(movey>=0):
            array[:, :movey] = 0
        else:
            array[:, movey:] = 0
        return array

def visualizeArray(Larray):
    nonzero = np.where(Larray[:, :]!=0)

    ax = plt.axes(projection='3d')
    ax.scatter3D(nonzero[0], nonzero[1], Larray[nonzero[0], nonzero[1]])
    plt.show()


class GFT:
    """This class provides utility functions to work with 
    ground map extracted from feet trajctories."""
    FeetTrajs = {}
    FeetTrajsDictList = {}
    ContactArray = np.empty(shape=0)
    GroundArray = np.empty(shape=0)
    GPMap = None
    Confidence = np.empty(shape=0)

    xlength = 0
    ylength = 0
    xNormal = 0
    yNormal = 0

    xRealRange = (0, 0)
    yRealRange = (0, 0)
    res = 0
    meanHeight = 0

    def __init__(self, GroundMapFile=None, FeetTrajsFile=None, InitializeGP = True) -> None:

        assert (GroundMapFile!=None and FeetTrajsFile==None) or (GroundMapFile==None and FeetTrajsFile!=None), \
            f"One and only one file should be provided"

        # Get GroundArray from saved GFT msgpack
        if GroundMapFile != None:
            self.GroundArray = self.load(GroundMapFile)
        else:
        # Get GroundArray from saved Feet Trajectories
            self.ContactArray, self.FeetTrajs = self.getContactPointsFromFile(
                FeetTrajsFile)
            self.GroundArray = self.getGroundFromContact(self.ContactArray)
            if(InitializeGP):
                self.initializeGPMap()

    def getContactPointsFromFile(self, FeetTrajsFile):
        """Get contact points from four feet Trajectories"""
        with open(FeetTrajsFile, 'rb') as data_file:
            data = data_file.read()
            data = msgpack.unpackb(data)
            print("File is loaded successfully.")

        self.FeetTrajsDictList = data

        FeetTrajs = {}
        for key, value in data.items():
            FeetTrajs[key] = np.array(value)

        ContactPoints = []
        for value in FeetTrajs.values():
            newPoints, _ = self.getContactPoints(value)
            ContactPoints = ContactPoints + newPoints

        return np.array(ContactPoints), FeetTrajs

    def getContactPoints(self, FootTrajSlice):
        """Get contact points from one foot's Trajectories. Construct big window and small window.
        Use the mean of the big window and extrame deviation to filter the points"""
        print("Extracting Contact points ......")
    # FootTrajSlice = FeetTrajs['LF_shank_fixed_LF_FOOT']
        ContactShow = []
        ContactPoints = []
        BIGWINDOW = 500
        SMALLWINDOW = 40
        SMALLWINDOWED = 0.015
        # Calculate the mean of the big window
        for bigWindowCount in range(len(FootTrajSlice)//BIGWINDOW+1):
            bWFront = bigWindowCount*BIGWINDOW
            bWBack = bWFront+BIGWINDOW
            if bWBack < len(FootTrajSlice) - 1:
                bWBack = bWBack
                ref = (FootTrajSlice[bWFront:bWBack, 2]).mean()
            else:
                # Do not update ref since the ref is not accurate
                bWBack = len(FootTrajSlice) - 1
            # print("Processing the "+str(bigWindowCount)+"th BigWindow")

            # Calculate the extrame deviation(ED) of the small window
            for smallWindowCount in range(bWBack - bWFront+1):
                sWFront = bWFront + smallWindowCount
                sWBack = sWFront + SMALLWINDOW
                if sWBack < len(FootTrajSlice) - 1:
                    sWBack = sWBack
                else:
                    # abondan last small window, since the ED is not accurate
                    break
                smallWindowData = FootTrajSlice[sWFront:sWBack, 2]
                # print(sWFront, sWBack)
                if abs(smallWindowData.max() - smallWindowData.min()) < SMALLWINDOWED and smallWindowData.min() < ref:
                    ContactShow.append(FootTrajSlice[sWFront, 2])
                    ContactPoints.append(FootTrajSlice[sWFront])
                    # ContactPoints.append(FootTrajSlice[sWFront+ SMALLWINDOW//2])
                else:
                    ContactShow.append(None)

        return ContactPoints, ContactShow

    def intArray(self, array):
        """Convert an numpy array or a number to int"""
        if(type(array) != np.ndarray):
            return math.floor(array)
        else:
            intarray = np.empty(shape=array.size, dtype=int)
            for i, num in enumerate(array):
                intarray[i] = math.floor(num)
            return intarray


    def getGroundFromContact(self, ContactArray, padding = True):
        """Get grid map of ground from Contact points. For each grid, calculate the mean of the contacts height. """
        self.res = 0.1
        self.xRealRange = (ContactArray[:, 0].min(), ContactArray[:, 0].max())
        self.yRealRange = (ContactArray[:, 1].min(), ContactArray[:, 1].max())
        self.xlength = math.ceil(
            (ContactArray[:, 0].max() - math.floor(ContactArray[:, 0].min()))/self.res)
        self.ylength = math.ceil(
            (ContactArray[:, 1].max() - math.floor(ContactArray[:, 1].min()))/self.res)
        self.xNormal = math.floor(ContactArray[:, 0].min()/self.res)
        self.yNormal = math.floor(ContactArray[:, 1].min()/self.res)

        x = self.intArray(ContactArray[:, 0]/self.res) - self.xNormal
        y = self.intArray(ContactArray[:, 1]/self.res) - self.yNormal
        z = ContactArray[:, 2]

        CountContactArray = np.zeros(shape=(self.xlength, self.ylength))
        GroundArray = np.zeros(shape=(self.xlength, self.ylength))
        for i in range(x.size):
            CountContactArray[x[i], y[i]] = CountContactArray[x[i], y[i]]+1
            GroundArray[x[i], y[i]] = GroundArray[x[i], y[i]] + z[i]
        GroundArray = np.true_divide(GroundArray, CountContactArray)

        GroundArray = np.nan_to_num(GroundArray)
        nonZero = np.where(GroundArray[:,:]!=0)
        self.meanHeight = (GroundArray[nonZero[0], nonZero[1]]).mean()

        if(padding):
            pad_width = 20
            GroundArray = np.pad(GroundArray, pad_width=pad_width, mode="constant", constant_values=0)
            self.xlength += 2*pad_width
            self.ylength += 2*pad_width
            self.xNormal -= pad_width
            self.yNormal -= pad_width
            self.xRealRange = (self.xRealRange[0] - 20*self.res, self.xRealRange[1] + 20*self.res)
            self.yRealRange = (self.yRealRange[0] - 20*self.res, self.yRealRange[1] + 20*self.res)

        return GroundArray

    def save(self, out_dir, GPMap=True, FeetTrajs = False):
        save_dict = {}
        save_dict["res"] = self.res
        save_dict["xNormal"] = self.xNormal
        save_dict["yNormal"] = self.yNormal
        save_dict["xRealRange"] = self.xRealRange
        save_dict["yRealRange"] = self.yRealRange
        save_dict["meanHeight"] = self.meanHeight
        save_dict["GroundArray"] = self.GroundArray.tolist()
        if(FeetTrajs):
            save_dict["FeetTrajs"] = self.FeetTrajsDictList
        if(GPMap):
            self.initializeGPMap()
            save_dict["GPMap"] = self.GPMap.tolist()
            save_dict["Confidence"] = self.Confidence.tolist()

        save_path = out_dir + "/GroundMap.msgpack"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as out_file:
            out_data = msgpack.packb(save_dict)
            out_file.write(out_data)

    def load(self, ground_dir, FeetTrajs = False, GPMap = True):
        with open(ground_dir, "rb") as data_file:
            data = data_file.read()
            ground_dict = msgpack.unpackb(data)
        self.res = ground_dict["res"]
        self.xNormal = ground_dict["xNormal"]
        self.yNormal = ground_dict["yNormal"]
        self.GroundArray = np.array(ground_dict["GroundArray"])
        self.xlength = self.GroundArray.shape[0]
        self.ylength = self.GroundArray.shape[1]
        self.xRealRange = ground_dict["xRealRange"]
        self.yRealRange = ground_dict["yRealRange"]
        self.meanHeight = ground_dict["meanHeight"]
        if(GPMap):
            self.GPMap = np.array(ground_dict["GPMap"])
            self.Confidence = np.array(ground_dict["Confidence"])
        if(FeetTrajs):
            self.FeetTrajsDictList = ground_dict["FeetTrajs"]

        for key, value in self.FeetTrajsDictList:
            self.FeetTrajs[key] = np.array(value)

        return self.GroundArray

    def visualizeOneFootTraj3D(self, range = None):
        """Visualize sparse grid map"""
        
        # z_nonzero = getGroundHeight(x_nonzero, y_nonzero)

        ax = plt.axes(projection='3d')
        plt.xlabel("x/m")
        plt.ylabel("y/m")
        ax.set_zlabel("Height/m")
        key = list(self.FeetTrajs.keys())[0]
        ax.scatter3D(self.FeetTrajs[key][:,0], self.FeetTrajs[key][:,1], self.FeetTrajs[key][:,2])

        plt.show()
    
    def visualizeContacts3D(self, range = None):
        """Visualize sparse grid map"""
        assert range is None or (range[-1]<self.xlength and range[-1]<self.ylength),\
            f"Range exceed the size of the map, the grid map size is {self.xlength} * {self.ylength}"
        nonzero = np.where(self.GroundArray[:, :]!=0)
        x_nonzero = nonzero[0][range]
        x_nonzeroReal = (x_nonzero+self.xNormal)*self.res

        y_nonzero = nonzero[1][range]
        y_nonzeroReal = (y_nonzero+self.yNormal)*self.res

        z_nonzero = self.GroundArray[x_nonzero, y_nonzero]
        # z_nonzero = getGroundHeight(x_nonzero, y_nonzero)

        ax = plt.axes(projection='3d')
        plt.xlabel("x/m")
        plt.ylabel("y/m")
        ax.set_zlabel("Height/m")

        ax.scatter3D(x_nonzeroReal, y_nonzeroReal, z_nonzero)

        plt.show()
        

    def visualizeContacts2D(self, range = None):
        assert range is None or (range[-1]<self.xlength and range[-1]<self.ylength),\
            f"Range exceed the size of the map, the grid map size is {self.xlength} * {self.ylength}"
        nonzero = np.where(self.GroundArray[:, :]!=0)
        x_nonzero = nonzero[0][range]
        x_nonzeroReal = (x_nonzero+self.xNormal)*self.res

        y_nonzero = nonzero[1][range]
        y_nonzeroReal = (y_nonzero+self.yNormal)*self.res
        # z_nonzero = getGroundHeight(x_nonzero, y_nonzero)

        plt.xlabel("x/m")
        plt.ylabel("y/m")

        plt.scatter(x_nonzeroReal, y_nonzeroReal)

        plt.show()

    def getHeight(self, x, y, method = 'sparse', visualize = False):
        # assert (all(self.xRealRange[0]<x) and all(x<self.xRealRange[1]) and all(self.yRealRange[0]<y) and all(y<self.yRealRange[1])), \
        #     f"the required data point is out of map, the map size is {self.xRealRange} * {self.yRealRange}"
        MSE = None
        if method == "sparse":
            height = self.GroundArray[self.intArray(x/self.res) - self.xNormal, self.intArray(y/self.res)-self.yNormal]
        
        elif method == "GP":
            xGround = self.intArray(x/self.res) - self.xNormal
            yGround = self.intArray(y/self.res)-self.yNormal            
            if type(xGround) == np.ndarray:
                height = []
                for i, j in zip(xGround, yGround):
                    height.append(self.getGPHeight(i, j, visualize = visualize))
                height = np.array(height)
            else:
                height, MSE = self.getGPHeight(xGround, yGround, visualize=visualize)
        
        elif method == "GPMap":
            height = self.GPMap[self.intArray(x/self.res) - self.xNormal, self.intArray(y/self.res)-self.yNormal]
            MSE = self.Confidence[self.intArray(x/self.res) - self.xNormal, self.intArray(y/self.res)-self.yNormal]
        return height, MSE

    def initializeGPMap(self, teststep = 10, trainstep = 50):
        """Reconsturct the map use Gaussian Process and contacts."""
        # Container
        self.GPMap = self.GroundArray.copy()
        self.Confidence = np.zeros(shape=self.GPMap.shape)
        GPMapCounter = np.zeros(shape=self.GPMap.shape)
        # Training set preparation
        occArray = self.GroundArray.copy()
        occArray[np.where(self.GroundArray[:, :]!=0)] = 1
        # Testing set (The points we want to estimate) preparation
        enhaceOccArray = occArray.copy()
        for movex in range(-teststep, teststep+1):
            for movey in range(-teststep, teststep+1): 
                enhaceOccArray = enhaceOccArray + moveArray(occArray.copy(), movex, movey)
        enhaceOccArray[np.where(enhaceOccArray[:, :]!=0)] = 1
        #Construct the windows
        xxBW, yyBW = np.meshgrid(np.arange(0,self.GroundArray.shape[0], trainstep - 2*teststep), np.arange(0,self.GroundArray.shape[1], trainstep - 2*teststep))
        xxBW = xxBW.flatten()
        yyBW = yyBW.flatten()
        BWS = np.stack([xxBW, yyBW]).T

        for xGround, yGround in BWS:
            # print(xGround, yGround)
            xLocalLength = min(xGround+trainstep, self.xlength) - xGround
            yLocalLength = min(yGround+trainstep, self.ylength) - yGround
            LocalArray = self.GroundArray[xGround: xGround+xLocalLength, yGround: yGround+yLocalLength]
            localTrain = np.where(LocalArray[:, :]!=0)
            localOcc = occArray[xGround: xGround+xLocalLength, yGround: yGround+yLocalLength]
            # visualizeArray(localOcc)
            # print(np.sum(localOcc))
            if(np.sum(localOcc)<=10):
                # print("No GP")
                continue
            else:
                # print("GP")
                prior = (LocalArray[localTrain[0], localTrain[1]]).mean()

                localEnhanceOcc = enhaceOccArray[xGround: xGround+xLocalLength, yGround: yGround+yLocalLength]
                localTest = np.where(localEnhanceOcc[:, :]!=0)
                

                trainX = np.stack([localTrain[0], localTrain[1]]).T
                trainY = (LocalArray[localTrain[0], localTrain[1]] - prior).reshape(-1, 1)

                kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], length_scale_bounds=(1e-2, 1e2))
                # kernel = C(1.0, (1e-3, 1e3)) * RBF([3,3], length_scale_bounds="fixed")
                # kernel =  C(1.0, (1e-3, 1e3)) * RBF([3,3], length_scale_bounds="fixed")


                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
                # print(gp.kernel.get_params())
                gp.fit(trainX, trainY)

                testX = np.stack([localTest[0], localTest[1]]).T
                testY, MSET = gp.predict(testX, return_std=True)
                zz = testY.T+prior
                self.GPMap[localTest[0] + xGround, localTest[1] + yGround] += zz[0]
                GPMapCounter[localTest[0] + xGround, localTest[1] + yGround] += 1
                self.Confidence[localTest[0] + xGround, localTest[1] + yGround] = MSET.T

        self.GPMap = np.true_divide(self.GPMap, GPMapCounter + occArray)

        self.Confidence[self.Confidence==0] = self.Confidence.max() + (self.Confidence.max() - self.Confidence.min())
        # self.Confidence = 1 - (self.Confidence - self.Confidence.min()) / (self.Confidence.max() - self.Confidence.min())

        # self.GPMap = np.nan_to_num(self.GPMap, nan=self.meanHeight)
        # visualizeArray(self.Confidence)
        # visualizeArray(self.GPMap)

    def getGPHeight(self, xGround, yGround, step = 20, visualize=False):
        """This function uses the contact points nearby to construct Gaussian Prosscess \
            to estimate the height of the target point"""

        if(max(xGround-step, 0)>= min(xGround+step, self.xlength) or max(yGround-step, 0)>= min(yGround+step, self.ylength)):
            return self.meanHeight, 0

        LocalArray = self.GroundArray[max(xGround-step, 0): min(xGround+step, self.xlength),\
                                max(yGround-step, 0): min(yGround+step, self.ylength)]
        nonzero = np.where(LocalArray[:, :]!=0)
        if(nonzero[0].size <=2):
            print("Warning: Unable to predict the point! Too less contacts nearby! Please check data or expend step!")
            return self.meanHeight, 0
        
        if(nonzero[0].size < 5):
            print("Warning: the number of points nearby is less than 5! ")
        
        trainX = np.stack([nonzero[0], nonzero[1]]).T
        prior = (LocalArray[nonzero[0], nonzero[1]]).mean()
        trainY = (LocalArray[nonzero[0], nonzero[1]] - prior).reshape(-1, 1)
        kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))
        # kernel = RBF([5,5], (1e-3, 1e3))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
        gp.fit(trainX, trainY)
        height, MSE = gp.predict(np.array([min(step, xGround), min(step, yGround)]).reshape(1, -1), return_std=True)
        height = height+prior

        if(visualize):
            xTest = np.arange(0, LocalArray.shape[0], 1)
            yTest = np.arange(0, LocalArray.shape[1], 1)
            # testX = np.stack([xTest, yTest])
            xx, yy = np.meshgrid(xTest, yTest)
            xx = xx.flatten()
            yy = yy.flatten()
            testX = np.stack([xx, yy])
            testY, vMSE = gp.predict(testX.T, return_std=True)
            zz = testY.T + prior
            ax = plt.axes(projection='3d')
            ax.scatter3D(nonzero[0], nonzero[1], LocalArray[nonzero[0], nonzero[1]], s = 30, c = 'r', label = "Sparse Grid Map")
            ax.scatter3D(xx, yy, zz, s = 10, c = "g", label = "GP Estimation")
            ax.scatter3D(min(step, xGround), min(step, yGround), height[0], s=60, c = 'b', label = "Target Point")
            ax.legend()
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Use GP to Estimate Height of One Position")
            ax.set_zlabel("Height/m")
            plt.show()
        
        return height[0], math.exp(-10*MSE[0])

    def getShape(self):
        return (self.xlength, self.ylength)

    def getMean(self, Array):
        return self.mean

    def visualizeGPMap(self):
        visualizeArray(self.GPMap)
    
    def getGPMap(self):
        return self.GPMap, self.Confidence
    
    def getSparseMap(self):
        return self.ContactArray


    def __sizeof__(self):
        return self.xlength*self.ylength


def main():
    # x = np.array([20, 20])
    # y = np.array([462, 462])
    # x = 20
    # y = 400
    # gft = GFT(FeetTrajsFile = '/home/anqiao/catkin_ws/SA_dataset/20211007_SA_Monkey_ANYmal_Chimera/chimera_mission_2021_10_10/mission1/Recontruct_2022-04-18-19-40-09_0/FeetTrajs.msgpack')
    # gft = GFT(FeetTrajsFile='/home/anqiao/catkin_ws/SA_dataset/20211007_SA_Monkey_ANYmal_Chimera/chimera_mission_2021_10_12/mission9/FeetTraj/Reconstruct-_2022-04-03-13-06-35_0/FeetTrajs.msgpack')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    gft = GFT(FeetTrajsFile='/home/anqiao/catkin_ws/SA_dataset/20211007_SA_Monkey_ANYmal_Chimera/chimera_mission_2021_10_08/mission3/WithPointCloudReconstruct_2022-04-01-21-41-55_0/FeetTrajs.msgpack', InitializeGP = True)
    # print(gft.getHeight(30.842474971923533,462.984496350972, method="GP"))
    gft.save('/home/anqiao/catkin_ws/SA_dataset/20211007_SA_Monkey_ANYmal_Chimera/chimera_mission_2021_10_08/mission3/WithPointCloudReconstruct_2022-04-01-21-41-55_0')
    # gft.save(dir_path+"/Example_Files", GPMap=True)
    
    # gft2 = GFT(GroundMapFile='/home/anqiao/catkin_ws/SA_dataset/20211007_SA_Monkey_ANYmal_Chimera/chimera_mission_2021_10_12/mission9/FeetTraj/Reconstruct-_2022-04-03-13-06-35_0/GroundMap.msgpack')
    # print(gft2.getHeight(34.842474971923533, 461.984496350972, method="GP", visualize=True))
    # gft.visualizeContacts3D()
    # gft.visualizeOneFootTraj3D()
    # gft.conver2GPMap()
    # gft2.visualizeGPMap()

    # print(gft2.getHeight(x, y, method="GP"))

    # gft2.visualizeContacts3D(range=np.arange(1, 100))


if __name__ == '__main__':
    main()
