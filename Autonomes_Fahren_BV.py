import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from mDev import *


GPIO.setmode(GPIO.BCM)					#It tells the library which pin nunbering system is being used

mdev = mDEV()							#Creates an object from the mDev file
mdev.writeReg(mdev.CMD_IO1,1)			#Red light
mdev.writeReg(mdev.CMD_IO2,1)			#Green light
mdev.writeReg(mdev.CMD_IO3,1)			#Blue light
mdev.writeReg(mdev.CMD_SERVO2,1500)		#Horizontal movement of the camera
mdev.writeReg(mdev.CMD_SERVO3,1250)		#Vertical movement of the camera


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FPS,30)		#Frame rate
capture.set(3,240)						#3 = Width of the frames in the video
capture.set(4,320)						#4 = Height of the frames  in the video

time.sleep(3)


#////////////////////  VIDEO FUNCTIONS  ////////////////////#


def maskingFrame(frame):																#Delimitates de area where the lines will be detected in a triangle
	region = np.array([[20,95],[-160,320],[410,320],[280,95]], dtype=np.int32)			#Array with the delimitation points
	mask = np.zeros_like(frame)															#Creates a black image
	cv2.fillPoly(mask, [region], 255)													#Draws a white trapezium on top of the black image
	maskedFrame = cv2.bitwise_and(frame, mask)											#Merges the black image with the trapezium
	return maskedFrame

def getLines(frame, lH, lS, lV, hH, hS, hV):																	#Function for finding lines on an image
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)																#Changes image from RGB-format to Hue Staturation
	LowerRegion = np.array([lH,lS,lV],np.uint8)																	#Lower limit for inRange function
	upperRegion = np.array([hH,hS,hV],np.uint8)																	#Upper limit for inRange function
	colorRange = cv2.inRange(hsv,LowerRegion,upperRegion)														#Marks as white pixels, all pixels inside the given range of values
	cannyFrame = cv2.Canny(colorRange, 50, 150)																	#Function used to detect the edges in an image
	croppedFrame = maskingFrame(cannyFrame)																		#Calls the function to delimitate the image
	cv2.imshow('filter', croppedFrame)																			#Displays the image with the applied filters
	lines = cv2.HoughLinesP(croppedFrame, 2, np.pi/180, 100, np.array([]), minLineLength=30, maxLineGap=30)		#Function that finds lines on the given image
	rightLine, leftLine = average(lines)																		#Calls the average function and returns the coordinates of two lines
	return rightLine, leftLine

def getObjects(frame, lH, lS, lV, hH, hS, hV):																	#Function that identifies objects of an especified color on an image
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)																#Changes image from RGB-format to Hue Staturation
	LowerRegion = np.array([lH,lS,lV],np.uint8)																	#Lower limit for inRange function
	upperRegion = np.array([hH,hS,hV],np.uint8)																	#Upper limit for inRange function
	colorRange = cv2.inRange(hsv,LowerRegion,upperRegion)														#Marks as white pixels, all pixels inside the given range of values
	kernel = np.ones((1,1),"uint8")																				#Creates a Kernel with a size of 1x1 pixels
	colorOb = cv2.morphologyEx(colorRange,cv2.MORPH_OPEN,kernel)												#Used for removing noise from the image
	colorOb = cv2.erode(colorOb, kernel, iterations = 5)														#Convolutes through the image and removes white noise
	colorOb = cv2.dilate(colorOb, kernel, iterations = 9)														#Increases the area of the detected ofbejt after erosion
	croppedFrame = maskingFrame(colorOb)																		#Calls the function to delimitate the image
	img, contours, hierarchy = cv2.findContours(croppedFrame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)	#Functions that finds all contours on a given image
	return contours

def coordinates(slope, intercept):					#Calculates the coordinates of the averaged lines
	y1 = 320										#Pixel value for y1
	y2 = int(y1*0.45)								#Pixel value for y2
	if slope == 0:									#Avoids error of dividing by 0
		return np.array([0,0,0,0])
	else:
		x1 = int((y1-intercept)/slope)				#Pixel value for x1
		x2 = int((y2-intercept)/slope)				#Pixel value for x2
		return np.array([x1, y1, x2, y2])			

def calc_avg(values):								#Function that calculates the average value of an array
	summ = 0
	for x in values:
		summ = summ + x								#Adds all the vaues of the array
	if len(values) == 0:							#Avoids error of dividing by 0
		avg = round(summ / 1, 2)
	else:
		avg = round(summ / len(values), 2)
	return avg

def average(lines):										#Creates a right and left line from a group of multiple lines
	rightslopes = []
	rightintercepts = []
	leftslopes = []
	leftintercepts = []
	rightLine = []
	leftLine = []
	lastL = []
	if lines is None:									#Avoids an exception error when no lines are found
		lines = lastL
	if lines is not None:
		for line in lines:								#Iterates through all the found lines
			lastL = line								
			x1, y1, x2, y2 = line.reshape(4)			#Divides the line (in array form) into four variables
			par = np.polyfit((x1,x2),(y1,y2), 1)		#Finds the slope and y-interception of the line using its coordinates
			slope = par[0]
			intercept = par[1]
			if slope > 0:								#If slope is positive then the line is considered as a rightline
				rightslopes.append(slope)
				rightintercepts.append(intercept)
			if slope < 0:								#If slope is negative then the line is considered as a leftline
				leftslopes.append(slope)
				leftintercepts.append(intercept)
		rightSlAv = calc_avg(rightslopes)				#Calls the calc_avg function and gets the average of the slopes
		rightInAv = calc_avg(rightintercepts)			#Calls the calc_avg function and gets the average of the y-interception
		leftSlAv = calc_avg(leftslopes)
		leftInAv = calc_avg(leftintercepts)
		rightLine = coordinates(rightSlAv, rightInAv)	#Uses the averaged interception and slope along with the coordinates function to make a new line
		leftLine = coordinates(leftSlAv, leftInAv)
	return rightLine, leftLine

def mergeFrameAndLines(frame, lines):								#Merges the right and left lines with the image from the camera
	linesOnFrame = np.zeros_like(frame)								#Creates a black image with the same size as the image from the camera
	if lines.size < 3 :												#If no complete line is detected in the array
		merged = cv2.addWeighted(frame, 0.8, linesOnFrame, 1, 1)	#Merges the black image and camera frame
		return merged
	if lines.size > 3:												#If a comlete line is detected in the array
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)						#Divides the line (in array form) into four variables 
			cv2.line(linesOnFrame, (x1,y1), (x2,y2), (0,0,255), 10)	#Draws the line on the black image
		merged = cv2.addWeighted(frame, 0.8, linesOnFrame, 1, 1)	#Merges the black image with lines and the camera frame
		return merged


#////////////////////  MOVEMENT FUNCTIONS  ////////////////////#


def forward():								#Vehicules moves forward in straight line
	global camHor
	mdev.writeReg(mdev.CMD_SERVO1,1500)		#Servomotor that dictates in which direction the vehicule turns (Right[1000] - Left[2000])
	mdev.writeReg(mdev.CMD_DIR1,1)			#Turning direction of the wheel1 (1 --> Forward / 0 --> Reverse)
	mdev.writeReg(mdev.CMD_DIR2,1)			#Turning direction of the wheel2 (1 --> Forward / 0 --> Reverse)
	mdev.writeReg(mdev.CMD_PWM1,speed)		#Turning speed of wheel1 (0 - 1000)
	mdev.writeReg(mdev.CMD_PWM2,speed)		#Turning speed of wheel2 (0 - 1000)

def goRight():								#Vehicule adjusts slightly to the right
	mdev.writeReg(mdev.CMD_SERVO1,1350)
	mdev.writeReg(mdev.CMD_DIR1,1)
	mdev.writeReg(mdev.CMD_DIR2,1)
	mdev.writeReg(mdev.CMD_PWM1,speed)
	mdev.writeReg(mdev.CMD_PWM2,speed)

def goLeft():								#Vehicule adjusts slightly to the left
	mdev.writeReg(mdev.CMD_SERVO1,1650)
	mdev.writeReg(mdev.CMD_DIR1,1)
	mdev.writeReg(mdev.CMD_DIR2,1)
	mdev.writeReg(mdev.CMD_PWM1,speed)
	mdev.writeReg(mdev.CMD_PWM2,speed)
    
def stop():									#Vehicule stops
	mdev.writeReg(mdev.CMD_SERVO1,1500)
	mdev.writeReg(mdev.CMD_DIR1,1)
	mdev.writeReg(mdev.CMD_DIR2,1)
	mdev.writeReg(mdev.CMD_PWM1,0)
	mdev.writeReg(mdev.CMD_PWM2,0)

def rightTurn():							#Vehicule makes a right turn
	mdev.writeReg(mdev.CMD_SERVO1,1050)
	mdev.writeReg(mdev.CMD_DIR1,1)
	mdev.writeReg(mdev.CMD_DIR2,1)
	mdev.writeReg(mdev.CMD_PWM1,speed)
	mdev.writeReg(mdev.CMD_PWM2,speed)

def leftTurn():								#Vehicule makes a left turn
	mdev.writeReg(mdev.CMD_SERVO1,1950)
	mdev.writeReg(mdev.CMD_DIR1,1)
	mdev.writeReg(mdev.CMD_DIR2,1)
	mdev.writeReg(mdev.CMD_PWM1,speed)
	mdev.writeReg(mdev.CMD_PWM2,speed)

def getDistance(trig,echo):				#Function for calculating distance with HC-SR04 Sensor
	GPIO.setup(trig, GPIO.OUT)			#Sets GPIO as output
	GPIO.setup(echo, GPIO.IN)			#Sets GPIO as input
	
	GPIO.output(trig, False)			#Output pin is set to low
	time.sleep(0.0002)					#Pause

	GPIO.output(trig, True)				#Output pin is set to high
	time.sleep(0.00001)					#Pause
	GPIO.output(trig, False)			#Output pin is set to low

	while GPIO.input(echo) == 0:		#Records time before the signal is recieved
		signal0 = time.time()

	while GPIO.input(echo) == 1:		#Records the time when the signal has been recieved
		signal1 = time.time()

	signalDuration = signal1 - signal0
	distance = signalDuration * 17150 	#Calculates the distance using time and speed of sound
	distance = round(distance, 2)		#Rounds the result to only 2 decimals
	GPIO.cleanup						#Resets the inputs and outputs
	return distance


#////////////////////  GLOBAL VARIABLES  ///////////////////#


camVer = 1250			#Camera´s vertical position
camHor = 1500			#Camera´s horizontal position
run = True				#Programm running when --> run = True
go = False				#Lets the vehicule advance or stop after detecting a green or red object
collision = False		#Lets vehicule know when to stop because of incomming collision
speed = 200				#Moving speed


#////////////////////  MAIN LOOP  /////////////////#


while run:
	_,vFrame = capture.read()											#Image from the camera is saved in a variable
	goFrame = vFrame.copy()												#Makes a copy of vFrame
	rightLine, leftLine = getLines(vFrame, 0, 0, 0, 179, 240, 120)		#Calls the function get lines using black color range
	blackLines = np.array([rightLine, leftLine], dtype="object")		#Creates an array containing the two averaged lines
	redContours = getObjects(vFrame, 0, 200, 30, 10, 255, 255)			#Finds the contour of red objects
	greenContours = getObjects(vFrame,25, 52, 72, 102, 255, 255)		#Finds the contour of green objects
	
	dist = getDistance(23,24)											#calls getDistance function using pin 23 as trig and pin 24 as echo
	print("frontal distance:  ", dist)
	if dist < 25:
		collision = True												#If frontal distance smaller than 25cm then collision = True
	elif dist > 25:
		collision = False												#If frontal distance bigger than 25cm then collision = False
	print("collision:  ", collision)
	xr, xl = 0, 0
	
	if go == False:
		for con in greenContours:										#Iterates through all the green contours
			if cv2.contourArea(con) > 300:								#If a contour with an area bigger than 300 pixels is found, go = True
				go = True
				cv2.drawContours(goFrame, con, -1, (0,0,255), 5)		#The contour is drawn on the goFrame
	if go == True:
		for con in redContours:											#Iterates through all the red contours
			if cv2.contourArea(con) > 300:								#If a contour with an area bigger than 300 pixels is found, go = False
				go = False
				cv2.drawContours(goFrame, con, -1, (0,255,0), 5)		#the contour is drawn on the goFrame
	print("go:   ", go)
	mergedFrame = mergeFrameAndLines(vFrame, blackLines)				#Lines and camera frame are merged into one image
	cv2.line(mergedFrame, (20,140), (20,200), (0,255,30), 5)
	cv2.line(mergedFrame, (80,140), (80,200), (0,255,30), 5)			#Lines are drawn on the camera image to delimitate the forward and turning moving areas
	cv2.line(mergedFrame, (210,140), (210,200), (50,200,0), 5)
	cv2.line(mergedFrame, (270,140), (270,200), (50,200,0), 5)
	
	if rightLine is not None:
		xr = blackLines[0][2]											#inner x-coodinate from right line
	if leftLine is not None:
		xl = blackLines[1][2]											#inner x-coordinate from left line
	cv2.line(mergedFrame, (xr,140), (xr,200), (255,50,0), 5)
	cv2.line(mergedFrame, (xl,140), (xl,200), (255,50,0), 5)			#This lines show how the edges of the road move in the image as the vehicule drives
	
	if go == True and collision == False:									#Green object was detected and no object is in front of the vehicule
		print(xl, " xl   ", xr, " xr")
		if (xl > 20 and xl < 80) and (xr > 210 and xr < 270):
			forward()														#Vehicule moves fordward
			print("going Forward")
		if (xl > 85 and xr > 275) and (xr == 0 and xl > 0 and xl < 165):
			goRight()														#Vehicule adjusts to the right
			print("adjusting Right")
		if (xr == 0 and xl > 180):
			rightTurn()														#Vehicule turns to the right
			print("turning Right")
		if (xl > 0 and xl < 15 and xr < 205) or (xl == 0 and xr > 165):
			goLeft()														#Vehicule adjusts to the left
			print("adjusting Left")
		if (xl == 0 and xr > 0 and xr < 180):
			leftTurn()														#Vehicule turns to the left
			print("turning Left")
	else:
		stop()																#Vehicule stops
		print("stop")
	
	cv2.imshow("Red & Green Contours", goFrame)							#Displays the image with the red and green object's contours
	cv2.imshow("Black Contours", mergedFrame)							#Displays the image with the road lines
	if cv2.waitKey(1) & 0xFF == ord('q'):								#Breaks the loop if the "q" key is pressed
		break

run = False
stop()
capture.release()														#Camera stops recording images
cv2.destroyAllWindows()													#All displayed images are eliminated

