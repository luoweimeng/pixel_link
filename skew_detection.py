# -*- coding: utf-8 -*-
#!/usr/bin/env python
#title           :
#description     :Judge whether the image should be corrected for skewness
#author          :luoweimeng
#date            :2018/7/4
#version         :
#usage           :
#notes           :
#python_version  :2.7 
#==============================================================================


import numpy as np
import cv2
import pixellink
import imutils

class skewDetector(object):
    def __init__(self, image_path):
        pl = pixellink.pixelLinkDetector(image_path)
        # 输出bounding boxes, (x1, y1, x2, y2, x3, y3, x4, y4) 顺时针方向
        pl.detect()

        # 输出pixel score

        pixel_score = pl.draw_pixel_score() * 256
        pixel_score = pixel_score.astype('int8')

        # 输出pixel score

        edged = cv2.Canny(np.uint8(pixel_score), 50, 256)
        # cv2.findContours(pixel_score_new.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]                # 用以区分OpenCV2.4和OpenCV3

        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:]
        self.rect = cv2.minAreaRect(np.concatenate(cnts))
        self.image = cv2.imread(image_path)
        self.image, _ = pixellink.resize_im(self.image, scale=768, max_scale=1280)

    def detect(self):
        '''
        Judge whether the image should be corrected for skewness
        :return: true if needed to be corrected, false if not
        '''
        return not self.rect[1][0] * self.rect[1][1] / (self.image.shape[0] * self.image.shape[1]) > 0.8

if __name__ == "__main__":
    sd = skewDetector("/Users/luoweimeng/Documents/较符合要求的图片/2.jpg")
    print(sd.detect())