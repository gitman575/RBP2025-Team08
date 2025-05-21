import numpy as np
import cv2

def color(t, v_min): #색 판별 (black, red, blue, green, other)
    h,s,v=t[0],t[1],t[2]
    if v<=v_min+3: return 1
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

def dfs(x, y, image, buffer): #모니터 찾기
    stack = []
    group = []
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

def color_detector(filename):
    global frame
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #HSV 변환
    H,W= len(img), len(img[0])
    monit = []
    for i in range(0,H,5):
        l=[0]*(W//5)
        monit.append(l)
    v_min=255
    for i in range(H//5):
        for j in range(W//5):
            if img[i*5][j*5][2]<v_min: v_min=img[i*5][j*5][2]

    for i in range(H//5):
        for j in range(W//5):
            t=img[i*5][j*5]
            monit[i][j]=color(t,v_min) #검은색 정의 후 1:5 크기로 축소, 색판별

    del img
    import gc
    gc.collect()

    buf = np.zeros_like(monit, dtype=np.uint8)
    f=0
    frame=[]
    for i in range(H//5):
        for j in range(W//5):
            if monit[i][j]==1:
                group = dfs(i,j,monit,buf)
                if len(group)>=len(frame):
                    frame=group #모니터
    '''
    disp=[]
    for i in range(H//5):
        l=[]
        for j in range(W//5):
            l.append([0]*3)
        disp.append(l)
    disp=np.array(disp)
    disp=disp.astype(np.uint8)
    for i in range(H//5):
        for j in range(W//5):
            if monit[i][j]==1: 
                disp[i][j]=[255, 255, 255]
                if (i,j) in frame: disp[i][j]=[0,255,255]
            elif monit[i][j]==2: disp[i][j]=[0,0,255]
            elif monit[i][j]==3: disp[i][j]=[0,255,0]
            elif monit[i][j]==4: disp[i][j]=[255,0,0]
    cv2.imshow('image',disp)
    cv2.waitKey(0)
    '''

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
    if m==G: return 'G'
    elif m==R: return 'R'
    else: return 'B'
    
if __name__ == '__main__':
    print(color_detector('input img name here'))
