# !/usr/bin/env python3
import rclpy
import cv2
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import numpy as np


memory = '0'
count = 0

def color_detector(img, i):
    global memory
    global count
#    img = cv2.imread(filename)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    
    #blurred = gray
    #lower_black = np.array([0,0,0])
    #upper_black = np.array([180,255,50])
    
    mask_black = cv2.inRange(blurred, 5, 40)
    kernel = np.ones((5, 5), np.uint8)
    mask_black = cv2.dilate(mask_black, kernel, iterations = i)

    #cv2.imshow('mask_black', mask_black)
    #cv2.waitKey(0)
    mask_black = cv2.bitwise_not(mask_black)

    _, bw = cv2.threshold(mask_black, 200, 0, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new = np.zeros(img.shape[:2], dtype = np.uint8)
    cv2.drawContours(new, contours, -1, 255, -1)
    #cv2.imshow('new', new)
    #cv2.waitKey(0)

    candidates = []
    
    

    for cnt in contours:
      area = cv2.contourArea(cnt)
      if img.shape[0]*img.shape[1]*0.1<area:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx)==4:
          candidates.append(cnt)

    if not candidates:
      count+=1
      if count<15:
        return memory
      else: 
        h_m = hsv[:,:,0]
        s_m = hsv[:,:,1]
        v_m = hsv[:,:,2]

        is_r = (((h_m < 20) | (h_m > 160)) & (s_m > 50) & (v_m > 50))
        is_b = ((h_m > 100) & (h_m < 140) & (s_m > 50) & (v_m > 50))
        cnt_r = np.sum(is_r)
        cnt_b = np.sum(is_b)
        return '-1' if cnt_r>=cnt_b and cnt_r>40000 else '+1' if cnt_b>cnt_r and cnt_b>40000 else '0'
      
    #print(candidates)
    largest_cnt = max(candidates, key = cv2.contourArea)
    epsilon = 0.01*cv2.arcLength(largest_cnt, True)
    approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
    # print(approx)
    # rect = cv2.minAreaRect(largest_cnt)
    # box = cv2.boxPoints(rect).astype(int)
    mask = np.zeros(img.shape[:2], dtype = np.uint8)
    cv2.drawContours(mask, [approx], -1, 255, thickness=-1)
    #cv2.imshow('new', new+mask)
    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)
    
    h_m = hsv[:,:,0]
    s_m = hsv[:,:,1]
    v_m = hsv[:,:,2]

    is_r = (((h_m < 20) | (h_m > 160)) & (s_m > 50) & (v_m > 50))&(mask == 255)
    is_b = ((h_m > 100) & (h_m < 140) & (s_m > 50) & (v_m > 50))&(mask == 255)
 
    is_o = (~(is_r | is_b) & (s_m>50) &(v_m>50))&(mask == 255)
    
    cnt_r = np.sum(is_r)
    cnt_b = np.sum(is_b)
    cnt_g = np.sum(is_o)


    color = '-1' if cnt_r>=cnt_g and cnt_r>=cnt_b else '0' if cnt_g>=cnt_b else '+1'
  
    memory = color
    count = 0
    return color 

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            # listen image topic
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            red_intensity=image[20, 10, 2]
            blue_intensity=image[20, 10, 0]

            # prepare rotate_cmd msg
            # DO NOT DELETE THE BELOW THREE LINES!
            msg = Header()
            msg = data.header
            msg.frame_id = '0'  # default: STOP
    
            # determine background color
            # TODO 
            msg.frame_id = color_detector(image, 1)
            # determine the color and assing +1, 0, or, -1 for frame_id
            # msg.frame_id = '+1' # CCW 
            # msg.frame_id = '0'  # STOP
            # msg.frame_id = '-1' # CW 
            
            #if red_intensity > blue_intensity:
            #    msg.frame_id='-1'
            #elif red_intensity < blue_intensity:
            #    msg.frame_id='+1'
            
            # publish color_state
            self.color_pub.publish(msg)
            
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % e)


if __name__ == '__main__':
    rclpy.init()
    detector = DetermineColor()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()
