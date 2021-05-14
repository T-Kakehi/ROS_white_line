#!/usr/bin/env python
# -*- coding: utf-8 -*-


import rospy
import numpy as np
from std_msgs.msg import Float64
from whiteline_py.msg import Image_Param
from geometry_msgs.msg import Twist

class ControlLane():
    def __init__(self):
        self.sub_lane = rospy.Subscriber('white_lane', Float64, self.cbFollowLane, queue_size = 1)
        self.sub_max_vel = rospy.Subscriber('max_vel', Float64, self.cbGetMaxVel, queue_size = 1)
        self.sub_image_param = rospy.Subscriber('img_param',Image_Param, self.GetImgParam, queue_size=1)
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size = 1)

        self.lastError = 0
        self.HEIGHT = 0
        self.WIDTH = 0
        self.MAX_VEL = 0.05

        rospy.on_shutdown(self.fnShutDown)

    def cbGetMaxVel(self, max_vel_msg):
        self.MAX_VEL = max_vel_msg.data

    def GetImgParam(self, img_param_msg):
        self.HEIGHT = img_param_msg.height
        self.WIDTH = img_param_msg.width
        #print(self.HEIGHT)
        #print(self.WIDTH)

    def cbFollowLane(self, desired_center):
        center = desired_center.data

        error = center - (self.WIDTH/2 - 15) #カメラと機体の中心の差分だけずれるので補正

        Kp = 0.002#0.0025 調整値
        Kd = 0.006#0.007 調整値

        # PID制御
        angular_z = Kp * error + Kd * (error - self.lastError)
        self.lastError = error
        twist = Twist()
        twist.linear.x = min(self.MAX_VEL * ((1 - abs(error) / (self.WIDTH / 2 -10)) ** 2.2), 0.2)
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
        self.pub_cmd_vel.publish(twist)

    def fnShutDown(self):
        rospy.loginfo("Shutting down. cmd_vel will be 0")

        twist = Twist()
        twist.linear.x = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = 0
        self.pub_cmd_vel.publish(twist) 

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('control_lane')
    node = ControlLane()
    node.main()
