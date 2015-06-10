# CV
import cv2
import cv2.cv as cv
import numpy as np
import time
from numpy import *

# Algorithms
from math import atan2, degrees, pi, radians, sqrt
from skimage import feature
from sklearn.svm import LinearSVC, SVC
from scipy.stats import mode

def cosd(angle):
    return cos(radians(angle))

def sind(angle):
    return sin(radians(angle))

class GaussianField():

    def __init__(self, height, width, scale=1, memory=5):
        self.queueSize = memory
        self.scale = scale
        self.height = height*scale
        self.width = width*scale
        self.accumArr = []

        for x in range(0, self.queueSize):
            self.accumArr.append([self.get_blank(), self.get_blank()])

    def get_blank(self):
        blank = ones((self.height,self.width), dtype=float64)
        blank = blank*0.0001
        return blank

    def add_object(self, scaleX, scaleY, Theta, x, y, sensor):

        # Theta
        if Theta > 90:
            Theta = 90 - Theta
        Theta = -Theta

        # Scale
        scaleX = scaleX/5
        scaleY = scaleY/5
        scaleX = scaleX*self.scale
        scaleY = scaleY*self.scale
        x = x*self.scale
        y = y*self.scale
        if scaleX == 0 or scaleY == 0:
            return

        # Scale and rotate bound
        r = 5
        rotTheta = abs(Theta)
        xRange = scaleY*r*sind(rotTheta) + scaleX*r*cosd(rotTheta)
        yRange = scaleX*r*sind(rotTheta) + scaleY*r*cosd(rotTheta)
        muX = 0
        muY = 0
        x = x - int(xRange/2)
        y = y - int(yRange/2)

        # Check out of bounds
        rightOut = (self.width) - (xRange+x)
        leftOut = x
        botOut = (self.height) - (yRange+y)
        topOut = y
        startY = y
        startX = x
        endY = yRange + y
        endX = xRange + x
        if rightOut < 0:
            endX = self.width
        else:
            rightOut = 0
        if leftOut < 0:
            startX = 0
        else:
            leftOut = 0 
        if botOut < 0:
            endY = self.height
        else:
            botOut = 0
        if topOut < 0:
            startY = 0
        else:
            topOut = 0 

        gauss_object = self.create_gaussian(xRange=xRange, yRange=yRange, scaleX=scaleX, scaleY=scaleY, Theta=Theta, A=1, muX=muX, muY=muY)
        gaussY, gaussX = gauss_object.shape
        gauss_object = gauss_object[-topOut:gaussY+botOut, -leftOut:gaussX+rightOut]
        gaussY, gaussX = gauss_object.shape
        blank = self.get_blank()

        try:
            diffX = endX - startX
            diffY = endY - startY
            xDiff = gaussX - diffX
            yDiff = gaussY - diffY
            blank[startY:endY+yDiff, startX:endX+xDiff] = gauss_object

            last_pane = self.accumArr[-1][sensor]
            last_pane = last_pane + blank
            last_pane = last_pane/linalg.norm(last_pane)
            last_pane *= 1.0/last_pane.max()
            self.accumArr[-1][sensor] = last_pane
        except Exception,e: 
            print str(e)

    def cycle(self):
        self.accumArr.pop(0)
        self.accumArr.append([self.get_blank(), self.get_blank()])

    def get_last_pane(self, sensor):
        pane = self.accumArr[-1][sensor]
        return cv2.resize(pane, (0,0), fx=1/self.scale, fy=1/self.scale) 

    def compute(self):
        blank = ones((self.height,self.width), dtype=float64)
        for x in range(self.queueSize-1, -1, -1):
            sensorData = self.accumArr[x]
            # This part seems slow
            for data in sensorData:
                blank = (blank*1)*data

        blank = blank/linalg.norm(blank)
        blank *= 1.0/blank.max()
        return cv2.resize(blank, (0,0), fx=1/self.scale, fy=1/self.scale) 

    def compute_and_process(self):
        F = self.compute()
        F = F*255
        F = F.astype(np.uint8)
        thresh = F.max() - 40
        ret,thresholdedImage = cv2.threshold(F,thresh,255,cv2.THRESH_BINARY)
        thresholdedImage = cv2.dilate(thresholdedImage,None,iterations = 3)
        thresholdedImage = cv2.erode(thresholdedImage,None,iterations = 3)
        ctr,heir = cv2.findContours(thresholdedImage,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        F = cv2.cvtColor(F, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(F,ctr,0,(0,255,255),2)

        frameArray = []
        for cnt in ctr:
            # Get centroid
            x,y,w,h = cv2.boundingRect(cnt)
            cx,cy = x+w/2, y+h/2   
            cv2.circle(F,(cx,cy), 2, (0,0,255), -1)    

            # Assign data to dictionary
            item={
                "Class": str(len(frameArray)),
                "Centroid": (cx, cy),
            }
            frameArray.append(item)

        return F, frameArray

    def create_gaussian(self, xRange, yRange, scaleX, scaleY, Theta, A, muX, muY):

        midX = int(xRange/2) 
        midY = int(yRange/2)
        x1 = arange(-midX,midX)
        x2 = arange(-midY,midY)
        X1, X2 = meshgrid(x1, x2)

        sigma1 = 1;
        sigma2 = 1;
        sigma1 = scaleX*sigma1;
        sigma2 = scaleY*sigma2;

        a = ((cosd(Theta)**2) / (2*sigma1**2)) + ((sind(Theta)**2) / (2*sigma2**2));
        b = -((sind(2*Theta)) / (4*sigma1**2)) + ((sind(2*Theta)) / (4*sigma2**2));
        c = ((sind(Theta)**2) / (2*sigma1**2)) + ((cosd(Theta)**2) / (2*sigma2**2));
        mu = [muX, muY];
        F = A*exp(-(a*(X1 - mu[0])**2 + 2*b*(X1 - mu[0])*(X2 - mu[1]) + c*(X2 - mu[1])**2));

        return F        
