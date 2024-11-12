import math
import numpy

from helper_functions import *

class Vehicle:
    def __init__(self, x, y, yaw, meters_to_pixels):
        #position and orientation of the vehicle
        self.x = x
        self.y = y
        self.yaw = yaw

        #convert meters to pixels
        self.meters_to_pixels = meters_to_pixels #how many pixels are in one meter?

        #vehicle dimensions in pixels
        self.length_px = 0.18 * self.meters_to_pixels
        self.width_px = 0.08 * self.meters_to_pixels

        #rear-center offset of the center of rear axle (where the black Dot on the vehicle is)
        self.rear_axle_offset_px = 0.065 * self.meters_to_pixels

        self.vehicle_speed = 0 #meters per second

        self.speed_filter_values = [] #history of speed values to filter

        self.last_update = 0
    
    def getPosition(self):
        return self.x, self.y
    
    def getOrientation(self):
        return self.yaw
    
    def getSpeed(self):
        return self.vehicle_speed
    
    def updatePose(self, x, y, yaw, current_time):
        dx = x - self.x
        dy = y - self.y
        dt = current_time - self.last_update
        dist = math.sqrt(dx**2 + dy**2)
        if dt != 0:
            speed = dist / dt
        else:
            speed = self.vehicle_speed

        #direction: forwards or backwards?
        angle_of_movement = getyaw( (self.x, self.y), (x, y) )
        forwards_motion = False
        if math.radians(-45) < angle_of_movement - yaw < math.radians(45):
            #movement along positive x axis of the vehicle
            forwards_motion = True
        
        if not forwards_motion:
            speed *= -1.0

        speed = speed / self.meters_to_pixels

        self.speed_filter_values.append(speed)
        self.speed_filter_values = self.speed_filter_values[-5:] #keep 5 most recent values

        self.x = x
        self.y = y
        self.vehicle_speed = sum(self.speed_filter_values) / len(self.speed_filter_values) #average
        self.yaw = yaw
        self.last_update = current_time
    
    #returns projected x,y,yaw for time at_time
    def getPositionEstimate(self, at_time):
        #TODO: implement constant velocity and turn rate model
        #for now simple linear estimation...
        dt = at_time - self.last_update

        distance = dt * self.vehicle_speed * self.meters_to_pixels
        x1 = self.x + math.cos(self.yaw) * distance
        y1 = self.y + math.sin(self.yaw) * distance

        return x1, y1, self.yaw
    
    #returns coordinates of a boundingbox of the vehicle.
    #at at_time is not None: projected into future.
    def getBoundingBox(self, at_time = None):
        pos_x, pos_y = self.getPosition()
        yaw = self.getOrientation
        if at_time is not None:
            pos_x, pos_y, yaw = self.getPositionEstimate(at_time)

        #define boundingbox in vehicle coordinate system (vehicle drives along x axis)
        bottom_left  = numpy.array([ -self.rear_axle_offset_px, int(-self.width_px/2) ])
        bottom_right = numpy.array([ -self.rear_axle_offset_px,  int(self.width_px/2) ])
        top_left     = numpy.array([ -self.rear_axle_offset_px+self.length_px, int(-self.width_px/2)])
        top_right    = numpy.array([ -self.rear_axle_offset_px+self.length_px,  int(self.width_px/2)])

        #rotate coordinates by yaw angle
        rotMatrix = numpy.array([[numpy.cos(yaw), -numpy.sin(yaw)], 
                         [numpy.sin(yaw),  numpy.cos(yaw)]])
        
        bottom_left = numpy.dot(rotMatrix, bottom_left)
        bottom_right = numpy.dot(rotMatrix, bottom_right)
        top_left = numpy.dot(rotMatrix, top_left)
        top_right = numpy.dot(rotMatrix, top_right)

        #move to x,y position
        bbox = [ list(bottom_left), list(bottom_right), list(top_right), list(top_left) ]
        for p in bbox:
            p[0] += pos_x
            p[1] += pos_y

        return bbox
    
