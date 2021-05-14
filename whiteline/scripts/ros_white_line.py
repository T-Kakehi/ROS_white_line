#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from whiteline_py.msg import Image_Param
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np

class WhiteLane():

    def __init__(self):
        self.sub_image = rospy.Subscriber("/usb_cam/image_raw", Image, self.process_whiteline, queue_size = 10)
        # self.sub_image = rospy.Subscriber("/webcam_0_camera/image_raw", Image, process_whiteline, queue_size = 1)
        self.pub_white_lane = rospy.Publisher('white_lane', Float64, queue_size = 1)
        self.pub_image_param = rospy.Publisher('img_param',Image_Param, queue_size=1)
        
        self.height = 480
        self.width = 640
        self.threshold = 200
        self.gap_lenge = 10
        self.slid_windows = 10

        self.top_width_rate=0.6
        self.top_height_position=0.4
        self.bottom_width_rate=1.2
        self.bottom_height_position=1

        self.top_left_position=0.5-(self.top_width_rate/2)
        self.top_right_position=0.5+(self.top_width_rate/2)
        self.bottom_left_position=0.5-(self.bottom_width_rate/2)
        self.bottom_right_position=0.5+(self.bottom_width_rate/2)

        self.bridge = CvBridge()

        #self.r = rospy.Rate(5)

        rospy.on_shutdown(self.fnShutDown)

    def Set_img_param(self):
        img_param = Image_Param()
        #print(self.height)
        #print(self.width)
        img_param.height = self.height
        img_param.width = self.width
        #rospy.loginfo(img_param.height)
        #rospy.loginfo(img_param.width)
        self.pub_image_param.publish(img_param)

    def process_whiteline(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg,"bgr8")
        self.Set_img_param()
        #cv2.imshow("src_img",img)
        #cv2.waitKey(100)
        img = cv2.resize(img, (self.width, self.height))
        try:
            # マスク領域の頂点を行列に型合わせ
            mask_points = [[self.width*self.top_left_position,self.height*self.top_height_position],[self.width*self.bottom_left_position,self.height*self.bottom_height_position],[self.width*self.bottom_right_position,self.height*self.bottom_height_position],[self.width*self.top_right_position,self.height*self.top_height_position]]
            vertices = np.array(mask_points).reshape((-1,1,2)).astype(np.int32)
            mask = np.zeros_like(img)
            if len(mask.shape)==2:
                cv2.fillPoly(mask, vertices, 255)
            else:
                cv2.fillPoly(mask, [vertices], (255,)*mask.shape[2])
            # cv2.imshow("mask_img",cv2.bitwise_and(img, mask))

            offset = self.width*.25

            ipm_points = [[self.width*self.top_left_position,self.height*self.top_height_position],[self.width*self.top_right_position,self.height*self.top_height_position],[self.width*self.bottom_right_position,self.height*self.bottom_height_position],[self.width*self.bottom_left_position,self.height*self.bottom_height_position]]
            src = np.float32(ipm_points)
            dst = np.float32([[offset, 0], [self.width - offset, 0], [self.width - offset, self.height], [offset, self.height]])

            # srcとdst座標に基づいて変換行列を作成する
            ipm_matrix = cv2.getPerspectiveTransform(src, dst)

            # 変換行列から画像をTopViewに変換する
            ipm_img = cv2.warpPerspective(img, ipm_matrix, (int(self.width), int(self.height)))

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
            ret, img_thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)

            # 二値化画像の表示
            #cv2.imshow("img_th", img_thresh)

            histogram = np.sum(img_thresh[int(self.height/2):,:], axis=0)
            # RGB変換
            cv_rgb_sliding_windows = np.dstack((img_thresh, img_thresh, img_thresh))

            midpoint = np.int(histogram.shape[0]/2)
            # windowのカレント位置を左右ヒストグラム最大となる位置で初期化する
            win_left_x = np.argmax(histogram[:midpoint])
            win_right_x = np.argmax(histogram[midpoint:]) + midpoint

            # window分割数を決める
            nwindows = int(self.height/self.slid_windows)

            # windowの高さを決める
            window_height = np.int(self.height/nwindows)
            # 画像内のすべての非ゼロピクセルのxとyの位置を特定する
            nonzero = img_thresh.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # window幅のマージン important
            margin = int(self.width/20)

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
                win_y_low = self.height - (window+1)*window_height
                win_y_high = self.height - window*window_height
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

            cv_rgb_sliding_windows[left_y, left_x] = [255, 0, 0]
            cv_rgb_sliding_windows[right_y, right_x] = [255, 0, 0]

            # 二次多項式
            left_equation = np.polyfit(left_y, left_x,2)
            right_equation = np.polyfit(right_y,right_x,2)
            #print(left_equation)
            #print(right_equation)

            center_equation = [(left_equation[0]+right_equation[0])/2,(left_equation[1]+right_equation[1])/2,(left_equation[2]+right_equation[2])/2]
            #print(center_equation)

            top_x_plot = (right_x[len(right_x)-1]+left_x[len(left_x)-1])/2
            center_x_plot = (right_x[int(len(right_x)/2)]+left_x[int(len(left_x)/2)])/2
            bottom_x_plot = (right_x[0]+left_x[0])/2

            #top_x_plot = int(center_equation[2])
            #center_x_plot = int((self.height/2)**center_equation[0]+(self.height/2)*center_equation[1]+center_equation[2])
            #bottom_x_plot = int(self.height**center_equation[0]+self.height*center_equation[1]+center_equation[2])
            
            cv2.line(cv_rgb_sliding_windows, (bottom_x_plot,self.height), (bottom_x_plot,int(self.height/2)),(0, 255, 0))
            cv2.arrowedLine(cv_rgb_sliding_windows, (bottom_x_plot,self.height), (center_x_plot,int(self.height/2)), (0, 0, 255), tipLength=0.05)
            cv2.line(cv_rgb_sliding_windows, (center_x_plot,int(self.height/2)), (center_x_plot,0), (0, 255, 0))
            cv2.arrowedLine(cv_rgb_sliding_windows, (center_x_plot,int(self.height/2)), (top_x_plot,0), (0, 0, 255), tipLength=0.05)

            cv2.imshow("sliding_view",cv_rgb_sliding_windows)
            cv2.waitKey(1)
            white_lane = Float64()
            white_lane.data = bottom_x_plot#center_x_plot
            self.pub_white_lane.publish(white_lane)

            #self.r.sleep()

        except Exception as err:
            pass

    def fnShutDown(self):
        rospy.loginfo("white_lane_node Shutting down.")

    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('white_lane')
    node = WhiteLane()
    node.main()