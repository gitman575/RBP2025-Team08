# !/usr/bin/env python3
import rclpy
import cv2
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import numpy as np

'''
def color(t, v_min): #색 판별 (black, red, blue, green, other)
    h,s,v=t[0],t[1],t[2]
    if v<=v_min+10: return 1
    if s>50 and v>50:
        if h>160 or h<20: return 2
        elif 40<h<80: return 3
        elif 100<h<140: return 4
    return 0

def check(x,y,image, buffer): #dfs용
    H, W=len(image), len(image[0])
    if 0<=x<H and 0<=y<W:
        if image[x][y]==1 and buffer[x][y]==0: return True
    return False

def dfs(x, y, image, buffer): #모니터 찾기기
    stack = []0
   0 group = []
    stack.append((x,y))
    while len(stack):
        X, Y = stack.pop()
        if buffer[X][Y]==1: continue
        group.append((X, Y))
        buffer[X][Y]=1;
        for i in range(-1,2):
            for j in range(-1,2):
                x_=X+i; y_=Y+j
                if check(x_,y_,image, buffer):
                    stack.append((x_,y_))
    return group

def color_detector(img):
    global frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #HSV 변환환
    H,W= len(img), len(img[0])
    monit = []
    for i in range(0,H):
        l=[0]*W
        monit.append(l)
    v_min=255
    for i in range(H):
        for j in range(W):
            if img[i][j][2]<v_min: v_min=img[i][j][2]

    for i in range(H):
        for j in range(W):
            t=img[i][j]
            monit[i][j]=color(t,v_min) #검은색 정의 후 1:5 크기로 축소, 색판별

    del img
    import gc
    gc.collect()

    buf = np.zeros_like(monit, dtype=np.uint8)
    f=0
    frame=[]
    for i in range(H):
        for j in range(W):
            if monit[i][j]==1:
                group = dfs(i,j,monit,buf)
                if len(group)>=len(frame):
                    frame=group #모니터
    
    disp=[]
    for i in range(H):
        l=[]
        for j in range(W):
            l.append([0]*3)
        disp.append(l)
    disp=np.array(disp)
    disp=disp.astype(np.uint8)
    for i in range(H):
        for j in range(W):
            if monit[i][j]==1: 
                disp[i][j]=[255, 255, 255]
                if (i,j) in frame: disp[i][j]=[0,255,255]
            elif monit[i][j]==2: disp[i][j]=[0,0,255]
            elif monit[i][j]==3: disp[i][j]=[0,255,0]
            elif monit[i][j]==4: disp[i][j]=[255,0,0]
    cv2.imshow('image',disp)
    cv2.waitKey(0)

    x_min = min(p[0] for p in frame)
    x_max = max(p[0] for p in frame)
    border = []
    l=[[] for i in range(x_min, x_max+1)]
    for i in frame:
        l[i[0]-x_min].append(i[1])
    for p in range(len(l)):
        y_min = min(l[p])
        y_max = max(l[p])
        border.append([y_min, y_max]) #모니터 내부
    
    del l
    gc.collect() #메모리 절약용

    R,G,B=0,0,0
    for i in range(x_min, x_max+1):
        r=border[i-x_min]
        for j in range(r[0], r[1]+1):
            if monit[i][j]==2: R+=1
            elif monit[i][j]==3: G+=1
            elif monit[i][j]==4: B+=1
    m=max(R,G,B)
    if m==G: return '0'
    elif m==R: return '-1'
    else: return '+1'
    '''
    


def color_detector(img):
#    img = cv2.imread(filename)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5),0)
    #lower_black = np.array([0,0,0])
    #upper_black = np.array([180,255,50])
    
    mask_black = cv2.inRange(blurred, 0, 30)
    kernel = np.ones((3, 3), np.uint8)
    mask_black = cv2.erode(mask_black, kernel, iterations=1)
    mask_black = cv2.dilate(mask_black, kernel, iterations=2)

    #cv2.imshow('mask_black', mask_black)
    #cv2.waitKey(0)

    _, bw = cv2.threshold(mask_black, 60, 255, cv2.THRESH_BINARY_INV)
   # cv2.imshow('bw', bw)
    contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
      area = cv2.contourArea(cnt)
      if area<img.shape[0]*img.shape[1]*0.95:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
          candidates.append(cnt)

    if not candidates:
      return "Fail"
    # print(candidates)
    largest_cnt = max(candidates, key = cv2.contourArea)
    epsilon = 0.01*cv2.arcLength(largest_cnt, True)
    approx = cv2.approxPolyDP(largest_cnt, epsilon, True)
    # print(approx)
    # rect = cv2.minAreaRect(largest_cnt)
    # box = cv2.boxPoints(rect).astype(int)
    mask = np.zeros(img.shape[:2], dtype = np.uint8)
    cv2.drawContours(mask, [approx], -1, 255, thickness=-1)
    #cv2.imshow('mask', mask)

    mask_r = (((hsv[:,:,0]>160)|(hsv[:,:,0]<20))&(hsv[:,:,1]>50)&(hsv[:,:,2]>50))
    mask_g = (((hsv[:,:,0]>40)&(hsv[:,:,0]<80))&(hsv[:,:,1]>50)&(hsv[:,:,2]>50))
    mask_b = ((hsv[:,:,0]>100)&(hsv[:,:,0]<140)&(hsv[:,:,1]>50)&(hsv[:,:,2]>50))

    cnt_r = int(np.count_nonzero(mask_r&(mask==255)))
    cnt_g = int(np.count_nonzero(mask_g&(mask==255)))
    cnt_b = int(np.count_nonzero(mask_b&(mask==255)))

    color = '-1' if cnt_r>=cnt_g and cnt_r>=cnt_b else '0' if cnt_g>=cnt_b else '+1'

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
            msg.frame_id = color_detector(image)
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
    
    

