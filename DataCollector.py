#토트 씨름로봇 색인식 코드 테스트 3가지 색상 구분 인식
#인식한 객체의 크기, 위치에 따라

import numpy as np
import cv2 as cv
import time

#------------------------------------------------------------------------------------------------------------------------------------------------

hsv1 = 0
lower_blue1 = 0
upper_blue1 = 0
lower_blue2 = 0
upper_blue2 = 0
lower_blue3 = 0
upper_blue3 = 0

hsv2 = 0
lower_green1 = 0
upper_green1 = 0
lower_green2 = 0
upper_green2 = 0
lower_green3 = 0
upper_green3 = 0

hsv3 = 0
lower_red1 = 0
upper_red1 = 0
lower_red2 = 0
upper_red2 = 0
lower_red3 = 0
upper_red3 = 0

start = 0 #딜레이 타이머

prevTime = time.time() #fps테스트

#hsv변환값 저장 + 마스킹 관련 값
#------------------------------------------------------------------------------------------------------------------------------------------------

def nothing(x): #트랙바 사용시 필요
    pass

def mouse_callback1(event, x, y, flags, param):
    global hsv_blue, lower_blue1, upper_blue1, lower_blue2, upper_blue2, lower_blue3, upper_blue3, threshold, start
    global hsv_green, lower_green1, upper_green1, lower_green2, upper_green2, lower_green3, upper_green3
    global hsv_red, lower_red1, upper_red1, lower_red2, upper_red2, lower_red3, upper_red3
    if event == cv.EVENT_LBUTTONDOWN: #마우스 좌 버튼 입력
        color = img_color1[y, x]
        one_pixel = np.uint8([[color]]) #자료형 변환
        hsv_blue = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV) #hsv로 색상공간 변경
        hsv_blue = hsv_blue[0][0] #각 차원에 담기는 정보?
        print(hsv_blue[0])
        threshold = cv.getTrackbarPos('threshold', 'img_result') #결과 win에서 threshold값 저장 변수
        File = open("blue.txt", "w")
        print("%d\n%d"%(hsv_blue[0],threshold), file=File)
        File.close()    

        # hsv 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 필셀값의 범위를 정합니다.
        if hsv_blue[0] < 10: #휴 값 범위에 따라 분류 처리 다르게 왜?
            # 마스크 3개씩
            lower_blue1 = np.array([hsv_blue[0]-10+180, threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv_blue[0], 255, 255])
            lower_blue3 = np.array([hsv_blue[0], threshold, threshold])
            upper_blue3 = np.array([hsv_blue[0]+10, 255, 255])

        elif hsv_blue[0] > 170:
            lower_blue1 = np.array([hsv_blue[0], threshold, threshold])
            upper_blue1 = np.array([180, 255, 255])
            lower_blue2 = np.array([0, threshold, threshold])
            upper_blue2 = np.array([hsv_blue[0]+10-180, 255, 255])
            lower_blue3 = np.array([hsv_blue[0]-10, threshold, threshold])
            upper_blue3 = np.array([hsv_blue[0], 255, 255])

        else:
            lower_blue1 = np.array([hsv_blue[0], threshold, threshold])
            upper_blue1 = np.array([hsv_blue[0]+10, 255, 255])
            lower_blue2 = np.array([hsv_blue[0]-10, threshold, threshold])
            upper_blue2 = np.array([hsv_blue[0], 255, 255])
            lower_blue3 = np.array([hsv_blue[0]-10, threshold, threshold])
            upper_blue3 = np.array([hsv_blue[0], 255, 255])

    if event == cv.EVENT_MBUTTONDOWN: #마우스 좌 버튼 입력
        color = img_color1[y, x]
        one_pixel = np.uint8([[color]]) #자료형 변환
        hsv_green = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV) #hsv로 색상공간 변경
        hsv_green = hsv_green[0][0] #각 차원에 담기는 정보?
        threshold = cv.getTrackbarPos('threshold', 'img_result') #결과 win에서 threshold값 저장 변수
        File = open("green.txt", "w")
        print("%d\n%d"%(hsv_green[0],threshold), file=File)
        File.close()    

        if hsv_green[0] < 10:
            # 마스크 3개씩
            lower_green1 = np.array([hsv_green[0]-10+180, threshold, threshold])
            upper_green1 = np.array([180, 255, 255])
            lower_green2 = np.array([0, threshold, threshold])
            upper_green2 = np.array([hsv_green[0], 255, 255])
            lower_green3 = np.array([hsv_green[0], threshold, threshold])
            upper_green3 = np.array([hsv_green[0]+10, 255, 255])

        elif hsv_green[0] > 170:
            lower_green1 = np.array([hsv_green[0], threshold, threshold])
            upper_green1 = np.array([180, 255, 255])
            lower_green2 = np.array([0, threshold, threshold])
            upper_green2 = np.array([hsv_green[0]+10-180, 255, 255])
            lower_green3 = np.array([hsv_green[0]-10, threshold, threshold])
            upper_green3 = np.array([hsv_green[0], 255, 255])

        else:
            lower_green1 = np.array([hsv_green[0], threshold, threshold])
            upper_green1 = np.array([hsv_green[0]+10, 255, 255])
            lower_green2 = np.array([hsv_green[0]-10, threshold, threshold])
            upper_green2 = np.array([hsv_green[0], 255, 255])
            lower_green3 = np.array([hsv_green[0]-10, threshold, threshold])
            upper_green3 = np.array([hsv_green[0], 255, 255])

    if event == cv.EVENT_RBUTTONDOWN: 
        color = img_color1[y, x]
        one_pixel = np.uint8([[color]]) 
        hsv_red = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv_red = hsv_red[0][0] 
        threshold = cv.getTrackbarPos('threshold', 'img_result')
        File = open("red.txt", "w")
        print("%d\n%d"%(hsv_red[0],threshold), file=File)
        File.close()

        if hsv_red[0] < 10: 
            lower_red1 = np.array([hsv_red[0]-10+180, threshold, threshold])
            upper_red1 = np.array([180, 255, 255])
            lower_red2 = np.array([0, threshold, threshold])
            upper_red2 = np.array([hsv_red[0], 255, 255])
            lower_red3 = np.array([hsv_red[0], threshold, threshold])
            upper_red3 = np.array([hsv_red[0]+10, 255, 255])

        elif hsv_red[0] > 170:
            lower_red1 = np.array([hsv_red[0], threshold, threshold])
            upper_red1 = np.array([180, 255, 255])
            lower_red2 = np.array([0, threshold, threshold])
            upper_red2 = np.array([hsv_red[0]+10-180, 255, 255])
            lower_red3 = np.array([hsv_red[0]-10, threshold, threshold])
            upper_red3 = np.array([hsv_red[0], 255, 255])

        else:
            lower_red1 = np.array([hsv_red[0], threshold, threshold])
            upper_red1 = np.array([hsv_red[0]+10, 255, 255])
            lower_red2 = np.array([hsv_red[0]-10, threshold, threshold])
            upper_red2 = np.array([hsv_red[0], 255, 255])
            lower_red3 = np.array([hsv_red[0]-10, threshold, threshold])
            upper_red3 = np.array([hsv_red[0], 255, 255])


cv.namedWindow('img_color1') #윈도우 창 생성
cv.setMouseCallback('img_color1', mouse_callback1) #콜백 매개 변수를 통해 다른 함수를 전달 받고, 이벤트가 발생할 때 매개 변수에 전달된 함수를 호출

cv.namedWindow('img_result') #윈도우 창 생성
cv.createTrackbar('threshold', 'img_result', 0, 255, nothing) #트랙바 생성 result window에
cv.setTrackbarPos('threshold', 'img_result', 30) #변환된 값이 저장될 변수, 생성 win, 초기 값

cap = cv.VideoCapture(0) #장치관리자 등록된 카메라 순서

#fps = cap.get(cv.CAP_PROP_FPS)

#--------------------------------------------------------------------------------------------------------------------------------------------

while(True):
    ret , img_color1 = cap.read() #비디오의 한 프레임씩 읽음 제대로 프레임을 읽으면 ret값이 True, 실패하면 False, fram에 읽은 프레임이 나옵니다 #shape 인덱스 0,1 = 높이, 폭
    img_color1 = cv.resize(img_color1, (400, 300), interpolation=cv.INTER_AREA)
    img_color1 = cv.bilateralFilter(img_color1, -1, 10, 5)
    for i in range(0,8):
        cv.line(img_color1, (i*50, 0), (i*50,300),(0,0,0))
    for i in range(0,6):
        cv.line(img_color1, (0, i*50), (400,i*50),(0,0,0))

#---------------------------------------fps테스트----------------------------------------------------------
    # 원본 영상을 hsv 영상으로 변환
    img_hsv1 = cv.cvtColor(img_color1, cv.COLOR_BGR2HSV) # BGR->hsv
    img_hsv1 = cv.bilateralFilter(img_hsv1, -1, 10, 5)

    # 범위 값으로 hsv 이미지에서 마스크를 생성, 관심 영역 설정

    img_Bmask1 = cv.inRange(img_hsv1, lower_blue1, upper_blue1)
    img_Bmask2 = cv.inRange(img_hsv1, lower_blue2, upper_blue2)
    img_Bmask3 = cv.inRange(img_hsv1, lower_blue3, upper_blue3)

    img_Gmask1 = cv.inRange(img_hsv1, lower_green1, upper_green1)
    img_Gmask2 = cv.inRange(img_hsv1, lower_green2, upper_green2)
    img_Gmask3 = cv.inRange(img_hsv1, lower_green3, upper_green3)

    img_Rmask1 = cv.inRange(img_hsv1, lower_red1, upper_red1)
    img_Rmask2 = cv.inRange(img_hsv1, lower_red2, upper_red2)
    img_Rmask3 = cv.inRange(img_hsv1, lower_red3, upper_red3)

    #cv.inRange => 입력행렬, 하한 값 행렬 스칼라, 상한 값 행렬 스칼라, 입력 영상과 같은 크기의 마스크 영상

    img_maskBlue = img_Bmask1 | img_Bmask2 | img_Bmask3 #하나의 마스크로 합연산
    img_maskGreen = img_Gmask1 | img_Gmask2 | img_Gmask3
    img_maskRed = img_Rmask1 | img_Rmask2 | img_Rmask3

    kernel = np.ones((11,11), np.uint8) # 커널 사이즈, Dilation 반대 작업 수행
    img_maskBlue = cv.morphologyEx(img_maskBlue, cv.MORPH_OPEN, kernel) #모폴로지 연산 노이즈 제거
    img_maskBlue = cv.morphologyEx(img_maskBlue, cv.MORPH_CLOSE, kernel)

    img_maskGreen = cv.morphologyEx(img_maskGreen, cv.MORPH_OPEN, kernel) 
    img_maskGreen = cv.morphologyEx(img_maskGreen, cv.MORPH_CLOSE, kernel)

    img_maskRed = cv.morphologyEx(img_maskRed, cv.MORPH_OPEN, kernel) 
    img_maskRed = cv.morphologyEx(img_maskRed, cv.MORPH_CLOSE, kernel)

    img_mask = img_maskBlue | img_maskGreen | img_maskRed

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득
    img_result = cv.bitwise_and(img_color1, img_color1, mask=img_mask) # 마스크 영역 겹치는 부분 출력

    numOfLabels1, img_label1, stats1, centroids1 = cv.connectedComponentsWithStats(img_maskBlue) # 트랙킹, 객체의 크기와 중심위치 함께 반환
    numOfLabels2, img_label2, stats2, centroids2 = cv.connectedComponentsWithStats(img_maskGreen)
    numOfLabels3, img_label3, stats3, centroids3 = cv.connectedComponentsWithStats(img_maskRed)
# image: 8비트 1채널 영상
# labels: 레이블 맵 행렬. 입력 영상과 같은 크기. numpy.ndarray.
# stats: 각 객체의 바운딩 박스, 픽셀 개수 정보를 담은 행렬. numpy.ndarray. shape=(N, 5), dtype=numpy.int32.
# centroids: 각 객체의 무게 중심 위치 정보를 담은 행렬 numpy.ndarray. shape=(N, 2), dtype=numpy.float64.
# ltype: labels 행렬 타입. cv2.CV_32S 또는 cv2.CV_16S. 기본값은 cv2.CV_32S
    
    for idx, centroid in enumerate(centroids1): #----------------------------------Blue------------------------------
        if stats1[idx][0] == 0 and stats1[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue
        
        x, y, width, height, area = stats1[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 50: #특정 크기 이상 객체만 트랙킹
            cv.circle(img_color1, (centerX, centerY), 10, (255,0,0), 10) #중심 위치
            cv.rectangle(img_color1, (x,y), (x+width, y+height), (255,0,0)) #크기 인식
            cv.putText(img_color1, "Blue",(x,y+30), cv.FONT_ITALIC, 1, (255,0,0), 1 )

#---------------------------------------------------------------------------------------------------------------------------------------------    
    for idx, centroid in enumerate(centroids2): #-------------------------------Green-----------------------------
        if stats2[idx][0] == 0 and stats2[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats2[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area<200: continue

        elif area >= 200: #특정 크기 이상 객체만 트랙킹
            cv.circle(img_color1, (centerX, centerY), 10, (0,255,0), 10) #중심 위치
            cv.rectangle(img_color1, (x,y), (x+width, y+height), (0,255,0)) #크기 인식
            cv.putText(img_color1, "Green",(x,y+30), cv.FONT_ITALIC, 1, (0,255,0), 1 )
            print("GreenArea :",area)
            
            if area > 5000:
                #if(start == 0):
                    #start = time.time()
                #else:
                    #if time.time()> start + 0.5:
                cv.putText(img_color1, "Battle_Mode",(x,y+100), cv.FONT_ITALIC, 1, (0,0,255), 1 )
                    #else:
                        #cv.putText(img_color1, "Normal_Mode",(x,y+100), cv.FONT_ITALIC, 1, (0,0,255), 1 )
            else:
                cv.putText(img_color1, "Normal_Mode",(x,y+100), cv.FONT_ITALIC, 1, (0,0,255), 1 )
                start = 0
#---------------------------------------------------------------------------------------------------------------------------------------------

    for idx, centroid in enumerate(centroids3): #----------------------------------Red------------------------------
        if stats3[idx][0] == 0 and stats3[idx][1] == 0:
            continue

        if np.any(np.isnan(centroid)):
            continue

        x, y, width, height, area = stats3[idx]
        centerX, centerY = int(centroid[0]), int(centroid[1])

        if area > 50: #특정 크기 이상 객체만 트랙킹
            cv.circle(img_color1, (centerX, centerY), 10, (0,0,255), 10) #중심 위치
            cv.rectangle(img_color1, (x,y), (x+width, y+height), (0,0,255)) #크기 인식
            cv.putText(img_color1, "Red",(x,y+30), cv.FONT_ITALIC, 1, (0,0,255), 1 )

    #img_mask = cv.bilateralFilter(img_mask,-1,10,5)

    cv.imshow('img_color1', img_color1)
    cv.imshow('img_mask', img_mask)
    cv.imshow('img_result', img_result)

    # ESC 키누르면 종료
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
