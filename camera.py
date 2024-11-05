import numpy as np
import cv2 as cv
import time

from helper_functions import *
from vehicle import Vehicle

class Camera:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        #configuration variables
        self.meters_to_pixels = 2000        #how many pixels are in one meter?
        self.frames_per_seconds = 30        #framerate of the camera
        self.max_speed_vehicle_mps = 4.0    #max speed of a car in meters per second
        self.minimum_brightness = 2.7       #used to brighten the image of the webcam
        self.threshold_brightness_of_black = 150
        self.threshold_brightness_of_white = 200
        self.size_between_black_and_white_center_px = 0.08 * self.meters_to_pixels
        self.height_over_ground_black_meters = 0.042
        self.height_over_ground_white_meters = 0.025
        self.circle_diameter_px = 0.02 * self.meters_to_pixels

        #camera properties
        self.opening_angle_vertical_degrees = 43.3
        self.opening_angle_horizontal_degrees = 70.42
        self.vertical_resolution_px = 1080
        self.horizontal_resolution_px = 1920
        


        #state tracked by the camera
        self.tracked_vehicles : list[Vehicle] = []  #vehicles found by the camera
        self.color_correct_ratio : float = 1.0
        self.current_time : float = 0

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
        y_angle_deg = (y_distance_from_center/ (self.vertical_resolution_px/2) ) * self.opening_angle_vertical_degrees/2


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


    
    def updateVehiclePosition(self, xPos : int, yPos : int, yaw : float, detect_new_vehicles=True):
        threshold = (self.max_speed_vehicle_mps * self.meters_to_pixels) / self.frames_per_seconds
        found_vehicle = None
        for v in self.tracked_vehicles:
            if distance( (xPos, yPos), list(v.getPosition()) ) < threshold:
                found_vehicle = v
                break
            else:
                print(f"Distance too large: {distance( (xPos, yPos), list(v.getPosition()) )} < {threshold}")
        
        if found_vehicle is not None:
            found_vehicle.updatePose(xPos, yPos, yaw, self.current_time)
        else:
            if detect_new_vehicles:
                vehicle = Vehicle(xPos, yPos, yaw)
                print(f"Found new Vehicle at {vehicle.getPosition()}, yaw angle {math.degrees(vehicle.getOrientation())}")
                self.tracked_vehicles.append(vehicle)



    def colorCorrectImage(self, frame, initializeRatio=False):
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

    def detectVehicles(self):
        # prepare object points for chessboard, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        #chessboard_n, chessboard_m = 9, 6
        #objp = np.zeros((chessboard_n*chessboard_m,3), np.float32)
        #objp[:,:2] = np.mgrid[0:chessboard_n,0:chessboard_m].T.reshape(-1,2)

        stopDetectionTime = time.time() + 1 # 1 seconds to detect vehicles

        while stopDetectionTime > time.time():
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            self.current_time = time.time()

            print(f"new frame at {self.current_time}-----------------------")
        
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
            circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT_ALT, 1.5, minDist=30, param1=300, param2=0.8, minRadius=int((self.circle_diameter_px/2.0)*0.9), maxRadius=int((self.circle_diameter_px/2.0)*1.1))
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
                            print(f"Distance between black and white circle: {distance(black, white)}")
                            #black and white match: we found a vehicle.
                            self.updateVehiclePosition(black[0], black[1], getyaw(black, white))

            # Display the resulting frame
            cv.imshow('frame', frame)
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

        margin_px = 40 #expand image by amount of pixels in any direction
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
        while len(self.tracked_vehicles) > 0:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            self.current_time = time.time()

            print("new frame -----------------------")
            time_start = time.time()
            
        
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            for vehicle in self.tracked_vehicles:

                updatedVehicle = False

                #estimate image boundaries where to search for the vehicle
                #there will probably be an offset which we need to track to convert back to full-image-coordinates
                boundingbox = vehicle.getBoundingBox(self.current_time)

                subimage, x_offset, y_offset = self.getSubimageFromPoints(boundingbox, frame)

                if subimage.shape[0] == 0 or subimage.shape[1] == 0:
                    #we lost a vehicle!
                    self.tracked_vehicles.remove(vehicle)
                    continue

                subimage_color = self.colorCorrectImage(subimage, initializeRatio=False)
                subimage_color = cv.medianBlur(subimage_color, 5)
                subimage_color = cv.medianBlur(subimage_color, 5)
                subimage_gray = cv.cvtColor(subimage_color, cv.COLOR_BGR2GRAY)

                # detect circles in the subimage
                circles = cv.HoughCircles(subimage_gray, cv.HOUGH_GRADIENT_ALT, 1.5, minDist=30, param1=300, param2=0.8, minRadius=int((self.circle_diameter_px/2.0)*0.8), maxRadius=int((self.circle_diameter_px/2.0)*1.2))
                
                # ensure at least some circles were found
                if circles is not None:
                    # convert the (x, y) coordinates and radius of the circles to integers
                    circles = np.round(circles[0, :]).astype("int")
                    print(f"Detected {len(circles)} circles.")
                    

                    black_dots, white_dots, subimage_color = self.classifyDotsFromCircles(circles, subimage_color)
                    print(f"Detected {len(black_dots)} black, and {len(white_dots)} white circles.")

                    #find a black circle which is close to a white one:
                    threshold = 0.3
                    for black in black_dots:
                        for white in white_dots:
                            if self.size_between_black_and_white_center_px * (1-threshold) < distance(black, white) < self.size_between_black_and_white_center_px * (1+threshold):
                                #black and white match: we found a vehicle.

                                #correct for height
                                black[0], black[1] = self.correctForHeightOfVehicle(black[0]+x_offset, black[1]+y_offset, self.height_over_ground_black_meters)
                                white[0], white[1] = self.correctForHeightOfVehicle(white[0]+x_offset, white[1]+y_offset, self.height_over_ground_white_meters)

                                

                                self.updateVehiclePosition(black[0], black[1], getyaw(black, white), detect_new_vehicles=False)
                                updatedVehicle = True
                            else:
                                print(f"too far apart: {distance(black, white)} px does not meet threshold {self.size_between_black_and_white_center_px}")
                
                """if not updatedVehicle:
                    print(f"did not find vehicle at {vehicle.getPosition()}, yaw: {vehicle.getOrientation()}")

                if updatedVehicle:
                    print(f"updated vehicle at {vehicle.getPosition()}, yaw: {vehicle.getOrientation()}")"""

            time_end = time.time()
            #debug info

            #show portion of image which we used
            frame[y_offset:y_offset+subimage_color.shape[0], x_offset:x_offset+subimage_color.shape[1]] = subimage_color

            for vehicle in self.tracked_vehicles:
                boundingbox = vehicle.getBoundingBox(self.current_time)
                self.drawBoundingBox(frame, boundingbox)
                cv.putText(frame, f"Speed: {vehicle.getSpeed():.2f} m/s", (int(boundingbox[0][0]), int(boundingbox[0][1])), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (0,0,0), 2, cv.LINE_AA)


            #print(f"Procesing time: {(time_end-time_start)*1000}ms")
            # Display the resulting frame
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break

        
                

        
        

if __name__ == "__main__":
    cam = Camera()
    while True:
        if cam.detectVehicles() > 0:
            cam.trackVehicles()
        else:
            print("Did not detect any vehicle.")