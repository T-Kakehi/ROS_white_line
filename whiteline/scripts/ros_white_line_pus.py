#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from whiteline_py.msg import Image_Param
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import math

height = 480
width = 640
threshold = 200
gap_lenge = 10

twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

def stop_motor():
    print("Motor STOP!!!")
    stop = Twist()
    stop.linear.x = 0
    stop.linear.y = 0
    stop.linear.z = 0
    stop.angular.x = 0
    stop.angular.y = 0
    stop.angular.z = 0
    twist_pub.publish(stop)

def process_whiteline(msg):
    img_pram_pub = rospy.Publisher('img_pram', Image_Param, queue_size=10)
    r = rospy.Rate(5)
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg,"bgr8")
    img = cv2.resize(img, (width, height))
    #cv2.imshow("src_img",img)
    try:
        top_width_rate=0.5
        top_height_position=0
        bottom_width_rate=1.0
        bottom_height_position=0.4

        top_left_position=0.5-(top_width_rate/2)
        top_right_position=0.5+(top_width_rate/2)
        bottom_left_position=0.5-(bottom_width_rate/2)
        bottom_right_position=0.5+(bottom_width_rate/2)

        img_pram = Image_Param()
        img_pram.height = height
        img_pram.width = width
        img_pram_pub.publish(img_pram)

        # マスク領域の頂点を行列に型合わせ
        mask_points = [[width*top_left_position,height*top_height_position],[width*bottom_left_position,height*bottom_height_position],[width*bottom_right_position,height*bottom_height_position],[width*top_right_position,height*top_height_position]]
        vertices = np.array(mask_points).reshape((-1,1,2)).astype(np.int32)
        mask = np.zeros_like(img)
        if len(mask.shape)==2:
            cv2.fillPoly(mask, vertices, 255)
        else:
            cv2.fillPoly(mask, [vertices], (255,)*mask.shape[2])
        #cv2.imshow("mask_img",cv2.bitwise_and(img, mask))

        offset = width*.25

        ipm_points = [[width*top_left_position,height*top_height_position],[width*top_right_position,height*top_height_position],[width*bottom_right_position,height*bottom_height_position],[width*bottom_left_position,height*bottom_height_position]]
        src = np.float32(ipm_points)
        dst = np.float32([[offset, 0], [width - offset, 0], [width - offset, height], [offset, height]])

        # srcとdst座標に基づいて変換行列を作成する
        ipm_matrix = cv2.getPerspectiveTransform(src, dst)

        # 変換行列から画像をTopViewに変換する
        ipm_img = cv2.warpPerspective(img, ipm_matrix, (int(width), int(height)))

        #cv2.imshow("topView",ipm_img)

        hsv = cv2.cvtColor(ipm_img,cv2.COLOR_BGR2HSV)
        frame = ipm_img.copy()

        # 白のHSV範囲
        lower_white = np.array([0,0,100])
        upper_white = np.array([180,50,255])

        # 白以外にマスク
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        res_white = cv2.bitwise_and(frame,frame, mask= mask_white)

        #cv2.imshow('white_line_src',res_white)

        # ガウスフィルタ
        gauss = cv2.GaussianBlur(res_white,(5,5),0)

        # 画像のグレイスケール化
        gray = cv2.cvtColor(gauss,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("gray",gray)

        # 二値化(閾値を超えた画素を255にする。)
        ret, img_thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # 二値化画像の表示
        #cv2.imshow("img_th", img_thresh)

        histogram = np.sum(img_thresh[int(height/2):,:], axis=0)
        # RGB変換
        cv_rgb_sliding_windows = np.dstack((img_thresh, img_thresh, img_thresh))

        midpoint = np.int(histogram.shape[0]/2)
        # windowのカレント位置を左右ヒストグラム最大となる位置で初期化する
        win_left_x = np.argmax(histogram[:midpoint])
        win_right_x = np.argmax(histogram[midpoint:]) + midpoint

        # window分割数を決める important
        nwindows = int(height/10)

        # windowの高さを決める
        window_height = np.int(height/nwindows)
        # 画像内のすべての非ゼロピクセルのxとyの位置を特定する
        nonzero = img_thresh.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # window幅のマージン important
        margin = int(width/20)

        # windowをセンタリングするための最小ピクセル数
        minpix = margin
        # 左右のレーンピクセルindexを持つための配列
        lane_left_idx = []
        lane_right_idx = []
        # windowの色
        rectangle_color=(0,160,0)

        last_lots_point=0 # 0: left, 1: right
        for window in range(nwindows):
            # 左右windowの座標を求める
            win_y_low = height - (window+1)*window_height
            win_y_high = height - window*window_height
            win_xleft_low = win_left_x - margin
            win_xleft_high = win_left_x + margin
            win_xright_low = win_right_x - margin
            win_xright_high = win_right_x + margin

            # 左右枠が被らないように調整する
            if win_xleft_high > win_xright_low:
                # 被っている
                over = win_xleft_high - win_xright_low
                if last_lots_point == 0:
                    # 左を優先する
                    win_xright_low = win_xleft_high
                    win_xright_high = win_xright_high + over
                    win_right_x = int((win_xright_low + win_xright_high)/2)
                else:
                    # 右を優先する
                    win_xleft_high = win_xright_low
                    win_xleft_low = win_xleft_low - over
                    win_left_x = int((win_xleft_low + win_xleft_high)/2)

            # 左右windowの枠を描画する
            cv2.rectangle(cv_rgb_sliding_windows,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),rectangle_color, 1)
            cv2.rectangle(cv_rgb_sliding_windows,(win_xright_low,win_y_low),(win_xright_high,win_y_high),rectangle_color, 1)

            # ウィンドウ内のxとyの非ゼロピクセルを取得する
            win_left_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            win_right_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # 枠内画素数をカウントする
            win_num_lefts = len(win_left_idx)
            win_num_rights = len(win_right_idx)
            
            # 次の開始位置は枠内要素が多い方を優先とする
            if win_num_lefts > win_num_rights:
                last_lots_point=0
            elif win_num_lefts < win_num_rights:
                last_lots_point=1
            else:
                # 要素数が同じ場合は直前の要素が多い方を優先として保持する
                pass

            # window内レーンピクセルを左右レーンピクセルに追加する
            lane_left_idx.append(win_left_idx)
            lane_right_idx.append(win_right_idx)

            # window開始x座標を更新する
            if win_num_lefts > minpix:
                last_win_left_x = win_left_x
                win_left_x = np.int(np.mean(nonzerox[win_left_idx]))
                # もし片方が空枠なら、次の開始位置を値のある枠と同じ量だけスライドする
                if win_num_rights == 0:
                    win_right_x += win_left_x - last_win_left_x
            if win_num_rights > minpix:
                last_win_right_x = win_right_x
                win_right_x = np.int(np.mean(nonzerox[win_right_idx]))
                # もし片方が空枠なら、次の開始位置を値のある枠と同じ量だけスライドする
                if win_num_lefts == 0:
                    win_left_x += win_right_x - last_win_right_x

        # window毎の配列を結合する
        lane_left_idx = np.concatenate(lane_left_idx)
        lane_right_idx = np.concatenate(lane_right_idx)

        # 左右レーンピクセル座標を取得するx
        left_x = nonzerox[lane_left_idx]
        left_y = nonzeroy[lane_left_idx]
        right_x = nonzerox[lane_right_idx]
        right_y = nonzeroy[lane_right_idx]

        high_color=200
        low_color=0
        cv_rgb_sliding_windows[left_y, left_x] = [high_color, low_color, low_color]
        cv_rgb_sliding_windows[right_y, right_x] = [low_color, low_color, high_color]

        cv2.imshow("sliding_view",cv_rgb_sliding_windows)

        # 二次多項式
        left_equation = np.polyfit(left_y, left_x,2)
        right_equation = np.polyfit(right_y,right_x,2)
        #print(left_equation)
        #print(right_equation)

        center_equation = [(left_equation[0]+right_equation[0])/2,(left_equation[1]+right_equation[1])/2,(left_equation[2]+right_equation[2])/2]
        #print(center_equation)

        y_plot = np.linspace(0,height-1,height)
        left_x_plot = y_plot**left_equation[0]+y_plot*left_equation[1]+left_equation[2]
        right_x_plot = y_plot**right_equation[0]+y_plot*right_equation[1]+right_equation[2]
        center_x_plot = y_plot**center_equation[0]+y_plot*center_equation[1]+center_equation[2]
        #plt.plot(left_x_plot,y)
        #plt.plot(right_x_plot,y)
        #plt.show()

        partition = 30
        # 下半分を分割
        partait = np.linspace(int(height/2),height,partition)
        # 一分割あたりのピクセル数
        partition_pts = int(height/2)/partition
        #print(partition_pts)
        center_x_partition_plot = partait**center_equation[0]+partait*center_equation[1]+center_equation[2]

        half_width = int(width/2)
        bias = half_width - center_x_partition_plot[0]
        print bias

        #print(center_x_partition_plot)
        tilt_list = [0,0,0,0,0]
        for i in range(0,10,2):
            gap = center_x_partition_plot[i+1] - center_x_partition_plot[i]
            #print(gap)
            tilt_list[i/2] = gap
        #print(tilt_list)
        print(tilt_list[0])
        #print("------------------")

        speed = 0.005

        twist = Twist()
        twist.linear.x = speed
        twist.linear.y = (tilt_list[0]*speed/partition_pts) + bias
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = (tilt_list[0]*speed/partition_pts)
        
        twist_pub.publish(twist)


        #print(type(center_x_partition_plot))

        #pts_left = np.int32(np.array([np.transpose(np.vstack([left_x_plot, y_plot]))]))
        #pts_right = np.int32(np.array([np.flipud(np.transpose(np.vstack([right_x_plot, y_plot])))]))
        #pts_center = np.int32(np.array([np.transpose(np.vstack([center_x_plot, y_plot]))]))
        #pts_center_partition = np.int32(np.array([np.transpose(np.vstack([center_x_partition_plot, partait]))]))

        #print(pts_left)
        #print(pts_right)
        #print(pts_center)
        #print(pts_center_partition)
        #print(type(pts_center_partition))

        cv2.waitKey(1)
        r.sleep()

    except Exception as err:
        cv2.imshow("src_img",img)
        #print err

def start_node():
    rospy.init_node('white_lane')
    rospy.loginfo('img_proc node started')
    rospy.Subscriber("/usb_cam/image_raw", Image, process_whiteline)
    onoff = rospy.Publisher('motor_power',Bool,queue_size=10)
    temp = Bool()
    temp.data = True
    onoff.publish(temp)

    print("debag start")
    rospy.spin()
    stop_motor()
    print("fin devag")


if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass