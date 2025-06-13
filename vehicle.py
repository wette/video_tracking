import math
import numpy
import socket
import struct

from helper_functions import *
from pid_controller import PIDController

class Vehicle:
    def __init__(self, x, y, yaw, meters_to_pixels):
        #position and orientation of the vehicle
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yaw_rate = 0.0
        
        #IP address
        self.port   = None
        self.IP     = None
        self.sock   = None

        #Offset Steering Servo
        self.servo_offset = 0.0
        self.max_steering_angle_rad = math.radians(45)

        #PID for speed control
        self.motor_pid = None
        self.current_motor_value = 0    #control signal to the hardware

        #reference to MPC contoroller
        self.mpc = None

        #color of the vehicle
        self.color = None

        #convert meters to pixels
        self.meters_to_pixels = meters_to_pixels #how many pixels are in one meter?

        #vehicle dimensions in pixels
        self.length_px = 0.18 * self.meters_to_pixels
        self.width_px = 0.08 * self.meters_to_pixels

        #rear-center offset of the center of rear axle (where the black Dot on the vehicle is)
        self.rear_axle_offset_px = 0.065 * self.meters_to_pixels

        #current speed of the vehicle
        self.vehicle_speed = 0 #meters per second

        self.speed_filter_values = [] #history of speed values to filter
        self.yaw_rate_filter_values = [] #history of yaw rate values to filter

        self.last_update = 0

        self.ttl = 5
    

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
        yaw_rate_rad_per_sec = 0.0
        dist = math.sqrt(dx**2 + dy**2)
        if dt != 0:
            speed = dist / dt
            yaw_rate_rad_per_sec = (yaw - self.yaw) / dt
        else:
            speed = self.vehicle_speed

        if speed < 0.05:
            speed = 0.0     #standstill

        #direction: forwards or backwards?
        angle_of_movement = getyaw( (self.x, self.y), (x, y) )
        forwards_motion = False
        if math.radians(-45) < angle_of_movement - yaw < math.radians(45):
            #movement along positive x axis of the vehicle
            forwards_motion = True
        
        if not forwards_motion:
            speed *= -1.0

        speed = speed / self.meters_to_pixels #convert to m/s

        self.speed_filter_values.append(speed)
        self.yaw_rate_filter_values.append(yaw_rate_rad_per_sec)
        self.speed_filter_values = self.speed_filter_values[-5:] #keep 5 most recent values
        self.yaw_rate_filter_values = self.yaw_rate_filter_values[-5:] #keep 5 most recent values

        self.x = x
        self.y = y
        self.vehicle_speed = sum(self.speed_filter_values) / len(self.speed_filter_values) #average
        self.yaw_rate = sum(self.yaw_rate_filter_values) / len(self.yaw_rate_filter_values) #average
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
    
    #initialize the PID controler used to control the speed of the vehicle
    def initMotorPID(self, kp:float, ki:float, kd:float, error_history_length:int):
        self.motor_pid = PIDController(kp, ki, kd, error_history_length)

    def initNetworkConnection(self, ip_adress:str, port:int):
        self.port = port
        self.IP = ip_adress
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # send controls via UDP to the vehicle hardware.
    # target_velocity_mps:          target velocity of the vehicle. in meters/second
    # target_steering_angle_rad:    target steering angle of the vehicle. in map coordiates.
    #                               positive is to the right
    def sendControlsToHardware(self, target_velocity_mps:float, target_steering_angle_rad:float):

        #ask PID control what to do with the motor voltage
        delta_motor_value = self.motor_pid.update(target_velocity_mps - self.vehicle_speed)
        self.current_motor_value += delta_motor_value
        self.current_motor_value = int(max(0, min(2**16, self.current_motor_value))) #clip between 0 and 2^16

        #compute delta steering angle
        target_steering_angle_rad = target_steering_angle_rad - self.yaw
        #check steering angle is in bounds
        target_steering_angle_rad = max(-self.max_steering_angle_rad, min(self.max_steering_angle_rad, target_steering_angle_rad))
        #convert steering angle to number between 0 and 2^16,
        steering_angle = int(((target_steering_angle_rad/self.max_steering_angle_rad + 1.0) / 0.5) * 2**16)

        #send data over to hardware
        msg = bytes()
        msg += struct.pack("!H", self.current_motor_value)    #network-byte-order unsigned 16bit int
        msg += struct.pack("!H", steering_angle)    #network-byte-order unsigned 16bit int
        
        self.sock.sendto(msg, (self.IP, self.port))
