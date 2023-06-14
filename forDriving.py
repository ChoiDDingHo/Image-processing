import numpy as np
import cv2 as cv
import time
import pigpio
import serial

cap = cv.VideoCapture(0, cv.CAP_V4L2)
#서보 모터 제어 세팅
pi = pigpio.pi()
servo_pin = 12
#시리얼 통신 세팅
ser = serial.Serial('/dev/ttyAMA1',9600)
message = ' '
gPos = 'N'
gHeight = 0
is_front = 0 #0 = 상대가 빨간원 뒤, 1 = 상대가 빨간원 앞
received_data =' '
dist = 255
trans_hex = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
#---------------------------------------------------------------------------------------------------------------------------
with open("green.txt", "r") as f:
    greens = f.readlines()
    greens = [line.rstrip('\n') for line in greens]
    greens = list(map(int, greens))
    
with open("red.txt", "r") as f:
    reds = f.readlines()
    reds = [line.rstrip('\n') for line in reds]
    reds = list(map(int, reds))
#-------------------------------------------------데이터 받아오기-----------------------------------------------------------
lower_green1 = np.array([greens[0], greens[1], greens[1]])
upper_green1 = np.array([greens[0]+10, 255, 255])
lower_green2 = np.array([greens[0]-10, greens[1], greens[1]])
upper_green2 = np.array([greens[0], 255, 255])

lower_red1 = np.array([reds[0], reds[1], reds[1]])
upper_red1 = np.array([reds[0]+10, 255, 255])
lower_red2 = np.array([reds[0]-10, reds[1], reds[1]])
upper_red2 = np.array([reds[0], 255, 255])

kernel = np.ones((7,7), np.uint8)
kernelHMT = np.array([[-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1],
                     [-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1],
                     [-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1]])

screen_width = 480
screen_height = 360
degreeConst = 15
#---------------------------------------------------------------------------------------------------------------------------
count = 0
sumfps = 0
avgfps = 0
prevTime = time.time()
#----------------------------------------------------------------------------------------------------------------------------
#teststart = time.time()
#----------------------------------------------------------------------------------------------------------------------------
running_degree = 1500 #각도 초기값 세팅

while(True):
    pi.set_servo_pulsewidth(servo_pin, running_degree)
    ret , img_color = cap.read() #비디오의 한 프레임씩 읽음 제대로 프레임을 읽으면 ret값이 True, 실패하면 False, fram에 읽은 프레임이 나옵니다
    #print("time1 :", time.time()-teststart)
    teststart= time.time()
    img_color = cv.resize(img_color, (screen_width, screen_height), interpolation=cv.INTER_AREA)
    #print("time1 :", time.time()-teststart)
    #teststart= time.time()

    img_hsv1 = cv.cvtColor(img_color, cv.COLOR_BGR2HSV) # BGR->hsv

    img_Gmask1 = cv.inRange(img_hsv1, lower_green1, upper_green1)
    img_Gmask2 = cv.inRange(img_hsv1, lower_green2, upper_green2)
    
    img_Rmask1 = cv.inRange(img_hsv1, lower_red1, upper_red1)
    img_Rmask2 = cv.inRange(img_hsv1, lower_red2, upper_red2)

    #print("time1 :", time.time()-teststart)
    #teststart= time.time()
#--------------------------------------------마스크---------------------------------------------------------
    img_maskGreen = img_Gmask1 | img_Gmask2
    
    img_maskRed = img_Rmask1 | img_Rmask2

    img_maskGreen = cv.morphologyEx(img_maskGreen, cv.MORPH_OPEN, kernel) 
    img_maskGreen = cv.morphologyEx(img_maskGreen, cv.MORPH_CLOSE, kernel)
    
    img_maskRed = cv.morphologyEx(img_maskRed, cv.MORPH_OPEN, kernel) 
    img_maskRed = cv.morphologyEx(img_maskRed, cv.MORPH_CLOSE, kernel)
    
    #img_maskRed = cv.morphologyEx(img_maskRed, cv.MORPH_HITMISS, kernelHMT) 
#------------------------------------------컨투어---------------------------------------------------------------
    numOfLabels1, img_label1, stats1, centroids1 = cv.connectedComponentsWithStats(img_maskGreen)
    numOfLabels2, img_label2, stats2, centroids2 = cv.connectedComponentsWithStats(img_maskRed)
    
    for idx, centroid in enumerate(centroids1): #---------------------------------Green-------------------------
        if stats1[idx][0] == 0 and stats1[idx][1] == 0 or np.any(np.isnan(centroid)):
            send_dist = 255
            if(gPos == 'R' and running_degree<2400): #right
                running_degree += 40
                
            elif(gPos == 'L' and running_degree>600): #left
                running_degree -= 40
    
    for idx, centroid in enumerate(centroids1): #---------------------------------Green-------------------------
        if stats1[idx][0] == 0 and stats1[idx][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        x, y, width, height, area = stats1[idx]  
        centerX, centerY = int(centroid[0]), int(centroid[1])
 
        if area < 100:
            gSize = '0'
            continue

        elif area >= 100: #특정 크기 이상 객체만 트랙킹 사이즈 6개 분할

            if(centerX>=240-70 and centerX<=240+70):
                gPos = 'C'
                    
            elif(centerX>240+70 and running_degree<2400): #right
                gPos = 'R'

            elif(centerX<240-70 and running_degree>600): #left
                gPos = 'L'
            
            if(height<=17):
                height = 18

            gHeight = float(np.log((height-17)/94))
            dist = (0.0325-((0.0325*0.0325)+4*0.0000532*(gHeight+0.00492))**(1/2))/(2*0.0000532)
            if dist < 0:
                dist = 0
            send_dist = int(dist)
            
            if(running_degree <=2400 and running_degree >=600 and gPos != 'C'):
                running_degree -= (centerX-(screen_width/2))/screen_width*degreeConst
                
    for idx, centroid in enumerate(centroids2): #---------------------------------Red-------------------------
        if stats2[idx][0] == 0 and stats2[idx][1] == 0:
            is_front = 0 #0이면 뒤, 1이면 앞
            continue
        if np.any(np.isnan(centroid)):
            is_front = 0
            continue
        x, y, width, height, area = stats2[idx]  
        if area<100:
            is_front = 0
            continue
        is_front = 1
        
#------------------------------------  ----------fps--------------------------------------------------------------
    curTime = time.time()
    fps = 1 / (curTime - prevTime)
    prevTime = curTime
    send_degree = int((running_degree-500)/2000 * 180)
    first_num = trans_hex[int(send_degree/16)]
    second_num = send_degree%16
    if(second_num>9):
        if(second_num==10): second_num = 'A'
        elif(second_num==11): second_num = 'B'
        elif(second_num==12): second_num = 'C'
        elif(second_num==13): second_num = 'D'
        elif(second_num==14): second_num = 'E'
        elif(second_num==15): second_num = 'F'
    
    send_degree = first_num + str(second_num)
    
    if np.any(np.isnan(gHeight)):
        continue
        
    if(send_dist>=255): send_dist = 255
    elif(send_dist<=0): send_dist = 0

    first_dist = trans_hex[int(send_dist/16)]
    second_dist = send_dist%16
    if(second_dist>9):
        if(second_dist==10): second_dist = 'A'
        elif(second_dist==11): second_dist = 'B'
        elif(second_dist==12): second_dist = 'C'
        elif(second_dist==13): second_dist = 'D'
        elif(second_dist==14): second_dist = 'E'
        elif(second_dist==15): second_dist = 'F'
        
    send_dist = first_dist + str(second_dist)
    
    message = '#' + send_degree + send_dist + '#'
    #time.sleep(0.5)
    ser.write(message.encode())
    #if ser.in_waiting > 0:
        #received_data = ser.readline().decode().rstrip() # 수신된 데이터 디코딩
        #print("nucleo :",received_data)

    if count == 60:
        count = 0
        avgfps = sumfps / 60
        print("FPS :",avgfps)
        print("distance(cm) :",int(dist))
        print("is front :",is_front)
        print("send_dist :",send_dist)
        avgfps = 0
        sumfps = 0
    else:
        sumfps += fps  
    count+=1
    
#------------------------------------------------------------------------------------------------------------
    cv.imshow('img_mask', img_maskGreen | img_maskRed)

    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
