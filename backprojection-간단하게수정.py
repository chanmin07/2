import cv2 as cv
import numpy as np
import math

# 모델 이미지와 HSV공간으로 변환한 모델 이미지
roi = cv.imread('model.png')
hsv_roi = cv.cvtColor(roi,cv.COLOR_BGR2HSV)

# 타겟 이미지와 HSV공간으로 변환한 타겟 이미지
target = cv.imread('target.jpg')
hsv_target = cv.cvtColor(target,cv.COLOR_BGR2HSV)
# 책에서 q단계로 줄인 2차원 히스토그램을 만든다. 여기서는 64를 사용하였다.
scale = 64 #높이면 선명도가 올라감

# 각각 q단계인 모델 HS히스토그램과 타겟 HS히스토그램을 만듬
model_hist = np.zeros((scale,scale))
target_hist = np.zeros((scale,scale))

# 알고리즘 2-2를 사용하여 정규화된 히스토그램을 만든다.(모델만 만듬.)
for i in range(hsv_roi.shape[1]):
    for j in range(hsv_roi.shape[0]):
        model_hist[math.trunc(hsv_roi.item(j,i,0)/180*(scale-1)),math.trunc(hsv_roi.item(j,i,1)*(scale-1)/255)]+=1
for i in range(scale):
    for j in range(scale):
        norm_model_hist = model_hist/(hsv_roi.shape[0] * hsv_roi.shape[1])
norm_model_hist /= np.max(norm_model_hist)

# 타겟이미지와 같은 크기를 가지는 빈 이미지를 만든다.
backP_img = np.zeros((target.shape[0],target.shape[1]),np.float64)
backP_img_u = np.zeros((target.shape[0],target.shape[1]),np.uint8)

# 타겟 이미지의 픽셀값을 양자화 하여 모델 히스토그램으로 이동하여 나오는 색상값으로 역투영을 수행한다.
for i in range(hsv_target.shape[1]):
    for j in range(hsv_target.shape[0]):
        backP_img[j,i] = norm_model_hist[math.trunc(hsv_target.item(j,i,0)*(scale-1)/180),math.trunc(hsv_target.item(j,i,1)*(scale-1)/255)]
        backP_img_u[j,i] = backP_img[j,i] * 255

# 이미지를 출력한다. imshow함수는 입력되는 배열의 값이 소수일 경우 [0.0, 1.0]의 범위를 [0, 255]에 매핑하여 변환해 출력해준다.
cv.imshow('img',backP_img)
cv.waitKey(0)
cv.destroyAllWindows()