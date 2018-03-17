import segmentImage
import SaliencyRC
import cv2
import numpy as np
import random
def test_segmentation():
    img3f = cv2.imread("test.jpg")
    img3f = img3f.astype(np.float32)
    img3f *= 1. / 255
    imgLab3f = cv2.cvtColor(img3f,cv2.COLOR_BGR2Lab)
    num,imgInd = segmentImage.SegmentImage(imgLab3f,None,0.5,200,50)

    print(num)
    print(imgInd)
    colors = [[random.randint(0,255),random.randint(0,255),random.randint(0,255)] for _ in range(num)]
    showImg = np.zeros(img3f.shape,dtype=np.int8)
    height = imgInd.shape[0]
    width = imgInd.shape[1]
    for y in range(height):
        for x in range(width):
            if imgInd[y,x].all() > 0:
                showImg[y,x] = colors[imgInd[y,x] % num]
    cv2.imshow("sb",showImg)
    cv2.waitKey(0)
def test_rc_map():
    img3i = cv2.imread("test.jpg")
    img3f = img3i.astype(np.float32)
    img3f *= 1. / 255
    #sal = SaliencyRC.GetRC(img3f,segK=20,segMinSize=200)
    start = cv2.getTickCount()
    sal = SaliencyRC.GetHC(img3f)
    end = cv2.getTickCount()
    print((end - start)/cv2.getTickFrequency())
    np.save("sal.npy",sal)
    idxs = np.where(sal < (sal.max()+sal.min()) / 1.8)
    img3i[idxs] = 0
    sal = sal * 255
    sal = sal.astype(np.int16)
    cv2.namedWindow("sb")
    cv2.moveWindow("sb",20,20)
    cv2.imshow('sb',sal.astype(np.int8))
    cv2.imshow("ss",img3i)
    cv2.waitKey(0)

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    #test_segmentation()
    test_rc_map()