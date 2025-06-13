import numpy as np
import cv2 as cv
import time

from helper_functions import *
from vehicle import Vehicle

class Camera:
    ColorMap = {"red"   : (1, 0, 0),
            "green" : (0, 1, 0),
            "blue"  : (0, 0, 1),
            "yellow" : (1, 1, 0),
            "orange" : (1, 0.5, 0)
            }
    
    def __init__(self,
                 vertical_resolution_px = 1080,
                 horizontal_resolution_px = 1920,
                 frames_per_seconds = 30,#90,
                 opening_angle_vertical_degrees = 88.0,
                 opening_angle_horizontal_degrees = 126.0,
                 meters_to_pixels = 666, #1075,
                 max_speed_vehicle_mps = 4.0,
                 minimum_brightness = 1.2, #2.7,
                 threshold_brightness_of_black = 150,
                 threshold_brightness_of_white = 200,
                 circle_diameter_meters = 0.023,
                 size_between_black_and_white_center_meters = 0.08,
                 height_over_ground_black_meters = 0.042,
                 height_over_ground_white_meters = 0.025,
                 cameraMatrix = np.array([[1.19164513e+03, 0.00000000e+00, 9.32255365e+02],
                             [0.00000000e+00, 1.19269246e+03, 5.44789222e+02],
                            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
                 distortionCoefficients = np.array([[ 0.02473071, -0.39668063,  0.00151336,  0.00085757,  0.25759047]])
                 ):
        #self.cap = cv.VideoCapture(0) #use default camera driver (might not support 90fps)
        #self.cap = cv.VideoCapture(0, cv.CAP_V4L) #use V4L to access the camera
        self.cap = cv.VideoCapture("/Users/wette/Documents/FHBielefeld/eigeneVorlesungen/AutonomeFahrzeuge1zu32/moving_vehicle.avi")
        
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        self.DEBUG = True

        #camera stream properties
        self.vertical_resolution_px = vertical_resolution_px
        self.horizontal_resolution_px = horizontal_resolution_px
        self.frames_per_seconds = frames_per_seconds
        self.opening_angle_vertical_degrees = opening_angle_vertical_degrees
        self.opening_angle_horizontal_degrees = opening_angle_horizontal_degrees
        
        #physical scenario: configuration variables
        self.meters_to_pixels = meters_to_pixels        #how many pixels are in one meter?
        self.max_speed_vehicle_mps = max_speed_vehicle_mps    #max speed of a car in meters per second
        self.minimum_brightness = minimum_brightness       #used to brighten the image of the webcam
        self.threshold_brightness_of_black = threshold_brightness_of_black
        self.threshold_brightness_of_white = threshold_brightness_of_white
        self.size_between_black_and_white_center_px = size_between_black_and_white_center_meters * self.meters_to_pixels #8cm from center to center
        self.height_over_ground_black_meters = height_over_ground_black_meters #how high is the black dot on the vehicle measured from the ground
        self.height_over_ground_white_meters = height_over_ground_white_meters #how high is the white dot on the vehicle measured from the ground
        self.circle_diameter_px = circle_diameter_meters * self.meters_to_pixels  #diameter of black and white dots (2cm)

        


        #intrinsic camera calibration
        #Camera matrix : 
        self.cameraMatrix = cameraMatrix

        #Distortion coefficients : 
        self.distortionCoefficients = distortionCoefficients

        #apply camera correction
        h,  w = self.horizontal_resolution_px, self.vertical_resolution_px
        newcameramtx, _ =cv.getOptimalNewCameraMatrix(self.cameraMatrix,self.distortionCoefficients,(w,h),1,(w,h))
        self.remapX, self.remapY = cv.initUndistortRectifyMap(self.cameraMatrix,self.distortionCoefficients, None, newcameramtx,(h, w),5)


        #state tracked by the camera
        self.tracked_vehicles : list[Vehicle] = []  #vehicles found by the camera
        self.color_correct_ratio : float = 1.0
        self.current_time : float = 0

        #for calculating FPS
        self.time_end = 0
        self.fps_buffer = []

        self.__setup_video_stream__()

    #configure video stream using V4L
    def __setup_video_stream__(self):
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G')) # depends on fourcc available camera
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.horizontal_resolution_px)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.vertical_resolution_px)
        self.cap.set(cv.CAP_PROP_FPS, self.frames_per_seconds)

    #destructor
    def __del__(self):
        # When everything done, release the capture
        self.cap.release()
        cv.destroyAllWindows()

    #project a point seen at a known height to the floor
    def correctForHeightOfVehicle(self, xPos: int, yPos: int, height_meters: float):
        #self.opening_angle_vertical_degrees = 43.3
        #self.opening_angle_horizontal_degrees = 70.42
        #self.vertical_resolution_px = 1920
        #self.horizontal_resolution_px = 1080

        #find out in which angle the object is seen
        #as an approximation: use linear interpolation
        x_distance_from_center = int( xPos - self.horizontal_resolution_px/2)
        y_distance_from_center = int( yPos - self.vertical_resolution_px/2)

        #linear interpolation
        x_angle_deg = (x_distance_from_center/ (self.horizontal_resolution_px/2) ) * self.opening_angle_horizontal_degrees/2
        y_angle_deg = (y_distance_from_center/ (self.vertical_resolution_px/2) )   * self.opening_angle_vertical_degrees/2


        #compute correction for height as projection into the floor
        tanx = math.tan(math.radians(90.0 - x_angle_deg))
        tany = math.tan(math.radians(90.0 - y_angle_deg))
        if tanx == 0:
            tanx = 0.00001
        if tany == 0:
            tany = 0.00001
        correction_x_meters = height_meters/tanx
        correction_y_meters = height_meters/tany

        #convert back to pixels
        correction_x_px = correction_x_meters * self.meters_to_pixels
        correction_y_px = correction_y_meters * self.meters_to_pixels

        #print(f"x, und y, winkel der Kamera: {x_angle_deg: .3f}, {y_angle_deg: .3f}")
        #print(f"x, und y, Korrektur: {correction_x_meters: .3f}, {correction_y_meters: .3f}")

        return xPos - correction_x_px, yPos - correction_y_px


    
    def updateVehiclePosition(self, xPos : int, yPos : int, yaw : float, color: list[int, int, int], detect_new_vehicles=True):
        threshold = (self.max_speed_vehicle_mps * self.meters_to_pixels) / self.frames_per_seconds
        found_vehicle = None
        for v in self.tracked_vehicles:
            if distance( (xPos, yPos), list(v.getPosition()) ) < threshold and \
                abs((v.getOrientation() - yaw)) < 0.3 or\
                abs((v.getOrientation() - yaw)) > (2*math.pi - 0.3):

                found_vehicle = v
                break
            else:
                if self.DEBUG: print(f"Distance too large or yaw not matching: {distance( (xPos, yPos), list(v.getPosition()) )} < {threshold}")
                if self.DEBUG: print(f"vehicle yaw: {v.getOrientation()}, circle yaw: {yaw}, diff: {v.getOrientation() - yaw}")
        
        if found_vehicle is not None:
            found_vehicle.updatePose(xPos, yPos, yaw, self.current_time)
            if self.DEBUG: print("Updating pose of vehicle to: ", xPos, yPos, yaw)
            return True
        else:
            if detect_new_vehicles:
                vehicle = Vehicle(xPos, yPos, yaw, self.meters_to_pixels)
                vehicle.color = color
                if self.DEBUG: print(f"Found new Vehicle at {vehicle.getPosition()}, yaw angle {math.degrees(vehicle.getOrientation())}")
                self.tracked_vehicles.append(vehicle)
                return True
        return False



    def colorCorrectImage(self, frame, initializeRatio=False):
        #return frame
        if initializeRatio:
            cols, rows, _ = frame.shape
            brightness = np.sum(frame) / (255 * cols * rows)
            
            self.color_correct_ratio = brightness / self.minimum_brightness
        frame = cv.convertScaleAbs(frame, alpha = 1 / self.color_correct_ratio, beta = 0)
        
        return frame
    
    def classifyDotsFromCircles(self, circles, frame):
        black_dots = []
        white_dots = []

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            
            if x < frame.shape[1] and y < frame.shape[0]:
                color = frame[y,x]

                #black
                thres = self.threshold_brightness_of_black
                if color[0] < thres and color[1] < thres and color[2] < thres:
                    #black dot found
                    black_dots.append( [x,y] )
                    cv.circle(frame, (x, y), r, (0, 0, 0), 4)
                    cv.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    #print(f"Found black Dot at {x}, {y}")
                
                #white
                thres = self.threshold_brightness_of_white
                if color[0] > thres and color[1] > thres and color[2] > thres:
                    #white dot found
                    white_dots.append( [x,y] )
                    cv.circle(frame, (x, y), r, (255, 255, 255), 4)
                    cv.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    #print(f"Found white Dot at {x}, {y}")
        return black_dots, white_dots, frame
    
    #figure out color of a vehicle at a given position
    def getColorOfVehicle(self, frame: cv.typing.MatLike, x: int, y: int, yaw: float) -> str:
        #assumption: black circle is surrounded by vehicle color.
        # hence, add a vector of size 1cm to the vehicle position (position of black circle) pointing in yaw direction
        x, y = rotate(x=0.01*self.meters_to_pixels + self.circle_diameter_px, y=0, alpha=yaw)
        if x < frame.shape[1] and y < frame.shape[0]:
                color = frame[y,x]
                minc = min(color)
                maxc = max(color)
                color = [(c-minc)/(maxc-minc) for c in color]   #map min-max to 0-1
                threshold = 0.3
                for name, val in Camera.ColorMap.items():
                    if color[0]-threshold < val[0] < color[0]+threshold and \
                        color[1]-threshold < val[1] < color[1]+threshold and \
                        color[2]-threshold < val[2] < color[2]+threshold:
                        return name
                    
                print("Error: Could not figure out which color the vehicle has: {color} at ({x},{y})")
                return None

        print("Error: Vehicle out of bounds: ({x},{y})")
        return None

    def detectVehicles(self):
        stopDetectionTime = time.time() + 1 # 1 seconds to detect vehicles

        while stopDetectionTime > time.time():
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            self.current_time = time.time()

            if self.DEBUG: print(f"new frame at {self.current_time}-----------------------")

            #apply camera correction to frame
            frame = cv.remap(frame, self.remapX, self.remapY, cv.INTER_LINEAR)
        
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            #color correct the image
            frame = self.colorCorrectImage(frame, initializeRatio=True)

            # Our operations on the frame come here
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.medianBlur(gray, 5)

            #find chessboard
            #ret, corners = cv.findChessboardCorners(gray, (chessboard_n, chessboard_m), None)
            #if ret:
            #    cv.drawChessboardCorners(frame, (chessboard_n, chessboard_m), corners, ret)


            # detect circles in the image
            #circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT_ALT, 1.5, minDist=30, param1=300, param2=0.8, minRadius=int((self.circle_diameter_px/2.0)*0.9), maxRadius=int((self.circle_diameter_px/2.0)*1.1))
            circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT_ALT, 1.5, minDist=20, param1=300, param2=0.8, minRadius=int((self.circle_diameter_px/2.0)*0.5), maxRadius=int((self.circle_diameter_px/2.0)*1.5))
            print(self.circle_diameter_px/2.0)
            # ensure at least some circles were found
            if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")

                black_dots, white_dots, frame = self.classifyDotsFromCircles(circles, frame)

                #find a black circle which is close to a white one:
                threshold = 0.3
                for black in black_dots:
                    for white in white_dots:
                        if self.size_between_black_and_white_center_px * (1-threshold) < distance(black, white) < self.size_between_black_and_white_center_px * (1+threshold):
                            if self.DEBUG: print(f"Distance between black and white circle: {distance(black, white)}")
                            #black and white match: we found a vehicle.
                            #color = self.getColorOfVehicle(frame, black[0], black[1], getyaw(black, white))

                            self.updateVehiclePosition(black[0], black[1], getyaw(black, white), color=None)
                        else:
                            if self.DEBUG: print(f"Distance too far between black and white circle: {distance(black, white)}")

            # Display the resulting frame
            cv.imshow('frame', frame)
            #cv.imshow('frame', gray)
            if cv.waitKey(1) == ord('q'):
                break

        return len(self.tracked_vehicles)

    #extract part of image which contains all points from points
    def getSubimageFromPoints(self, points, frame):
        xs = []
        ys = []
        for p in points:
            xs.append(p[0])
            ys.append(p[1])

        margin_px = 10 #expand image by amount of pixels in any direction
        xstart = int(max(min(xs)-margin_px, 0))
        xend   = int(min(max(xs)+margin_px, frame.shape[1]))
        ystart = int(max(min(ys)-margin_px, 0))
        yend   = int(min(max(ys)+margin_px, frame.shape[0]))

        return frame[ystart:yend, xstart:xend], xstart, ystart

    def drawBoundingBox(self, frame, boundingbox):
        for i in range(len(boundingbox)):
            from_pt = boundingbox[i]
            to_pt = boundingbox[0]
            if i + 1 < len(boundingbox):
                to_pt = boundingbox[i+1]
                
            cv.line(frame, [int(x) for x in from_pt], [int(x) for x in to_pt], (255, 0, 0))

    def trackVehicles(self):

    
        # Capture frame-by-frame
        ret, frame = self.cap.read()
        self.current_time = time.time()

        if self.DEBUG: print("new frame -----------------------")

        #apply camera correction to frame
        frame = cv.remap(frame, self.remapX, self.remapY, cv.INTER_LINEAR)
        
        time_for_one_pass = -1
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return

        for vehicle in self.tracked_vehicles:

            updatedVehicle = False

            #estimate image boundaries where to search for the vehicle
            #there will probably be an offset which we need to track to convert back to full-image-coordinates
            boundingbox = vehicle.getBoundingBox(self.current_time)

            subimage, x_offset, y_offset = self.getSubimageFromPoints(boundingbox, frame)

            if subimage.shape[0] < 2 or subimage.shape[1] < 2:
                #we lost a vehicle!
                self.tracked_vehicles.remove(vehicle)
                continue


            #subimage_color = self.colorCorrectImage(subimage, initializeRatio=False)
            subimage_color = subimage
            #subimage_color = subimage

            subimage_color = cv.medianBlur(subimage_color, 5)
            #subimage_color = cv.medianBlur(subimage_color, 5) #2nd time blur
            subimage_gray = cv.cvtColor(subimage_color, cv.COLOR_BGR2GRAY)

            # detect circles in the subimage
            circles = cv.HoughCircles(subimage_gray, cv.HOUGH_GRADIENT_ALT, 1.5, minDist=20, param1=300, param2=0.8, minRadius=int((self.circle_diameter_px/2.0)*0.5), maxRadius=int((self.circle_diameter_px/2.0)*1.5))
            
            # ensure at least some circles were found
            if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
                if self.DEBUG: print(f"Detected {len(circles)} circles.")
                if self.DEBUG and len(circles) < 2:
                    print(f"cv.HoughCircles(subimage_gray, cv.HOUGH_GRADIENT_ALT, 1.5, minDist=30, param1=300, param2=0.8, minRadius={int((self.circle_diameter_px/2.0)*0.5)}, maxRadius={int((self.circle_diameter_px/2.0)*1.5)})")
                

                #black_dots, white_dots, subimage_color = self.classifyDotsFromCircles(circles, subimage_color)
                #if self.DEBUG: print(f"Detected {len(black_dots)} black, and {len(white_dots)} white circles.")

                #find a black circle which is close to a white one:
                threshold = 0.5
                for dot1 in circles:
                    cv.circle(subimage_color, (dot1[0], dot1[1]), dot1[2], (0, 0, 0), 4)
                    for dot2 in circles:
                        if dot1[0] == dot2[0] and dot1[1] == dot2[1] :
                            continue
                        minDist = self.size_between_black_and_white_center_px * (1-threshold)
                        maxDist = self.size_between_black_and_white_center_px * (1+threshold)
                        if minDist < distance(dot1, dot2) < maxDist:
                            #black and white match: we found a vehicle.
                            color1 = subimage_gray[dot1[1],dot1[0]]
                            color2 = subimage_gray[dot2[1],dot2[0]]

                            if abs(color1 -color2) < 30:
                                #two circles of same color
                                continue

                            black = dot1.copy()
                            white = dot2.copy()
                            if color1 > color2:
                                black = dot2.copy()
                                white = dot1.copy()

                            #correct for height
                            black[0], black[1] = self.correctForHeightOfVehicle(black[0]+x_offset, black[1]+y_offset, self.height_over_ground_black_meters)
                            white[0], white[1] = self.correctForHeightOfVehicle(white[0]+x_offset, white[1]+y_offset, self.height_over_ground_white_meters)

                            
                            if not updatedVehicle:
                                updatedVehicle = self.updateVehiclePosition(black[0], black[1], getyaw(black, white), color=None, detect_new_vehicles=False)
                        else:
                            if self.DEBUG: print(f"too far apart: {distance(dot1, dot2)} px does not meet threshold {self.size_between_black_and_white_center_px}")
            
            if not updatedVehicle:
                print(f"did not find vehicle at {vehicle.getPosition()}, yaw: {vehicle.getOrientation()}")
                #we lost a vehicle!
                vehicle.ttl -= 1
                if vehicle.ttl < 0:
                    self.tracked_vehicles.remove(vehicle)
                    continue
            else:
                vehicle.ttl = 15


            """if updatedVehicle:
                print(f"updated vehicle at {vehicle.getPosition()}, yaw: {vehicle.getOrientation()}")"""


            #show portion of image which we used
            frame[y_offset:y_offset+subimage_color.shape[0], x_offset:x_offset+subimage_color.shape[1]] = subimage_color

            time_for_one_pass = time.time() - self.time_end
            self.time_end = time.time()
            #debug info


        for vehicle in self.tracked_vehicles:
            boundingbox = vehicle.getBoundingBox(self.current_time)
            self.drawBoundingBox(frame, boundingbox)
            cv.putText(frame, f"Speed: {vehicle.getSpeed():.2f} m/s", (int(boundingbox[0][0]), int(boundingbox[0][1])), cv.FONT_HERSHEY_SIMPLEX, 1,
                        (0,255,0), 2, cv.LINE_AA)

        #print FPS
        if time_for_one_pass > 0.0:
            fps = 1.0/time_for_one_pass
            self.fps_buffer.append(fps)
            self.fps_buffer = self.fps_buffer[-self.frames_per_seconds:len(self.fps_buffer)] #keep last second

        cv.putText(frame, f"FPS: {sum(self.fps_buffer)/len(self.fps_buffer)}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1,
                        (255,0,0), 2, cv.LINE_AA)

        #print(f"Procesing time: {(time_end-time_start)*1000}ms")
        # Display the resulting frame
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            return

        

        

if __name__ == "__main__":

    cam = Camera()
    while True:
        if cam.detectVehicles() > 0:
            while len(cam.tracked_vehicles) > 0:
                cam.trackVehicles()
            while cv.waitKey(1) != ord('n'):
                pass
        else:
            print("Lost track - restarting.")
