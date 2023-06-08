import numpy as np
import cv2 as cv
import time
import serial

cap = cv.VideoCapture(0)

#---------------------------------------------------------------------------------------------------------------------------
#ser = serial.Serial('COM3',9600, timeout = 1)
#ser.flushInput()
message = 'N'
gPos = 'N'
gSize = 'N'
bPos = 'N'
rPos = 'N'
received_data = ''
trans_hex=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
#---------------------------------------------------------------------------------------------------------------------------
with open("blue.txt", "r") as f:
    blues = f.readlines()
    blues = [line.rstrip('\n') for line in blues]
    blues = list(map(int, blues))

with open("green.txt", "r") as f:
    greens = f.readlines()
    greens = [line.rstrip('\n') for line in greens]
    greens = list(map(int, greens))

with open("red.txt", "r") as f:
    reds = f.readlines()
    reds = [line.rstrip('\n') for line in reds]
    reds = list(map(int, reds))
#-------------------------------------------------데이터 받아오기-----------------------------------------------------------
lower_blue1 = np.array([blues[0]-5, blues[1], blues[1]])
upper_blue1 = np.array([blues[0]+5, 255, 255])

lower_green1 = np.array([greens[0], greens[1], greens[1]])
upper_green1 = np.array([greens[0]+10, 255, 255])
lower_green2 = np.array([greens[0]-5, greens[1], greens[1]])
upper_green2 = np.array([greens[0]+5, 255, 255])
lower_green3 = np.array([greens[0]-10, greens[1], greens[1]])
upper_green3 = np.array([greens[0], 255, 255])

lower_red1 = np.array([reds[0]-10, reds[1], reds[1]])
upper_red1 = np.array([reds[0], 255, 255])
lower_red2 = np.array([reds[0], reds[1], reds[1]])
upper_red2 = np.array([reds[0]+10, 255, 255])

kernel = np.ones((5,5), np.uint8)
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
    ret , img_color = cap.read() #비디오의 한 프레임씩 읽음 제대로 프레임을 읽으면 ret값이 True, 실패하면 False, fram에 읽은 프레임이 나옵니다
    #print("time1 :", time.time()-teststart)
    teststart= time.time()
    img_color = cv.resize(img_color, (400, 300), interpolation=cv.INTER_AREA)
    #print("time1 :", time.time()-teststart)
    #teststart= time.time()

    img_hsv1 = cv.cvtColor(img_color, cv.COLOR_BGR2HSV) # BGR->hsv

    img_Bmask1 = cv.inRange(img_hsv1, lower_blue1, upper_blue1)

    img_Gmask1 = cv.inRange(img_hsv1, lower_green1, upper_green1)
    img_Gmask2 = cv.inRange(img_hsv1, lower_green2, upper_green2)
    img_Gmask3 = cv.inRange(img_hsv1, lower_green3, upper_green3)

    img_Rmask1 = cv.inRange(img_hsv1, lower_red1, upper_red1)
    img_Rmask2 = cv.inRange(img_hsv1, lower_red2, upper_red2)

    #print("time1 :", time.time()-teststart)
    #teststart= time.time()
#--------------------------------------------마스크---------------------------------------------------------
    img_maskGreen = img_Gmask1 | img_Gmask2 | img_Gmask3
    img_maskRed = img_Rmask1 | img_Rmask2

    img_Bmask1 = cv.morphologyEx(img_Bmask1, cv.MORPH_OPEN, kernel) #모폴로지 연산 노이즈 제거
    img_Bmask1 = cv.morphologyEx(img_Bmask1, cv.MORPH_CLOSE, kernel)

    img_maskGreen = cv.morphologyEx(img_maskGreen, cv.MORPH_OPEN, kernel) 
    img_maskGreen = cv.morphologyEx(img_maskGreen, cv.MORPH_CLOSE, kernel)

    img_maskRed = cv.morphologyEx(img_maskRed, cv.MORPH_OPEN, kernel) 
    img_maskRed = cv.morphologyEx(img_maskRed, cv.MORPH_CLOSE, kernel)

    img_mask = img_Bmask1 | img_maskGreen | img_maskRed
#------------------------------------------컨투어---------------------------------------------------------------
    numOfLabels1, img_label1, stats1, centroids1 = cv.connectedComponentsWithStats(img_Bmask1)
    numOfLabels1, img_label1, stats2, centroids2 = cv.connectedComponentsWithStats(img_maskGreen)
    numOfLabels1, img_label1, stats3, centroids3 = cv.connectedComponentsWithStats(img_maskRed)

    for idx, centroid in enumerate(centroids1): #---------------------------------blue--------------------------
        if stats1[idx][0] == 0 and stats1[idx][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        x, y, width, height, area = stats1[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1]) 
        if area > 50: #특정 크기 이상 객체만 트랙킹
            if area > 120: #특정 크기 이상 객체만 트랙킹
                if(centerX>=160 and centerX<=240):
                    bPos = 'C'
                    
                elif(centerX>240): #right
                    bPos = 'R'
                    
                elif(centerX<160): #left
                    bPos = 'L'
    
    for idx, centroid in enumerate(centroids2): #----------------------Green시야 각 벗어남-----------------------
        if stats2[idx][0] == 0 and stats2[idx][1] == 0:
            if(gPos == 'R' and running_degree<2400): #right
                running_degree += 50
                
            elif(gPos == 'L' and running_degree>600): #left
                running_degree -= 50
    
    for idx, centroid in enumerate(centroids2): #---------------------------------Green-------------------------
        if stats2[idx][0] == 0 and stats2[idx][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        x, y, width, height, area = stats2[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])
        if area < 200: 
            continue

        elif area >= 200: #특정 크기 이상 객체만 트랙킹 사이즈 6개 분할
            if area > 3000:
                gSize = 'B'
            elif area > 2000:
                gSize = 'b'
            elif area > 1500:
                gSize = 'M'
            elif area > 1000:
                gSize = 'm'
            elif area > 500:
                gSize = 'S'
            else:
                gSize = 's'

            if(centerX>=180 and centerX<=220):
                gPos = 'C'
                    
            elif(centerX>=320): #right
                running_degree += 30
                gPos = 'R'

            elif(centerX<320 and centerX>220): #right
                running_degree += 15
                gPos = 'r'

            elif(centerX<=80): #left
                running_degree -= 30
                gPos = 'L'

            elif(centerX>80 and centerX<180): #right
                running_degree -= 15
                gPos = 'l'

    for idx, centroid in enumerate(centroids3): #---------------------------------Red----------------------------
        if stats3[idx][0] == 0 and stats3[idx][1] == 0:
            continue
        if np.any(np.isnan(centroid)):
            continue
        x, y, width, height, area = stats3[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])
        if area > 120: #특정 크기 이상 객체만 트랙킹
            if(centerX>=160 and centerX<=240):
                rPos = 'C'
                    
            elif(centerX>240): #right
                rPos = 'R'
                    
            elif(centerX<160): #left
                rPos = 'L'

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
    print(send_degree)
    message = '#'+send_degree+gSize+bPos+rPos+'#'
    print("pi :",message)
    #ser.write(message.encode())
    #if ser.in_waiting > 0:
        #received_data = ser.readline().decode().rstrip() # 수신된 데이터 디코딩
        #print("nucleo :",received_data)


    if count%10 == 0:
        if count == 60:
            count = 0
            avgfps = sumfps / 60
            print("FPS :",avgfps)
            avgfps = 0
            sumfps = 0
    else:
        sumfps += fps  
    count+=1
    
#------------------------------------------------------------------------------------------------------------
    cv.imshow('img_mask', img_mask)

    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
