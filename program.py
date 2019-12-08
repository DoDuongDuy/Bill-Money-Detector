import cv2
import numpy as np 
import math
import glob
from matplotlib import pyplot as plt 
money_template = []
money_value = []
def load_template():
    files = glob.glob("template\\*.png")
    for item in files:
        img = cv2.imread(item, 0)
        money_template.append (img)
        val = item.split("\\")[-1]
        val = val.split(".")[0]
        money_value.append(val)
def get_contour_large(img):
    img_blur = cv2.blur(img, (5,5))
    img_canny = cv2.Canny(img_blur,30, 40)
    _, img_threshold = cv2.threshold(img_canny, 0, 255, cv2.THRESH_OTSU)
    kernel = np.ones((30,30),np.uint8)
    img_closed = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel)
    _, contours, _ = cv2.findContours(img_closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnt = None
    s_max_large = 0
    for item in contours:
        s = cv2.contourArea(item)
        if(s > s_max_large):
            s_max_large = s
            cnt = item
    return cnt
def money_detector(img):
    cnt = get_contour_large(img)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    dist1 = int(math.sqrt((box[1][0]  - box[0][0])**2 + (box[1][1]  - box[0][1])**2))
    dist2 = int(math.sqrt((box[3][0]  - box[0][0])**2 + (box[3][1]  - box[0][1])**2))
    angle = rect[2]
    ratio = 1
    if(dist1 > dist2):
        angle += 90
        ratio = 500/dist1
    else:
        ratio = 500/dist2
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cx,cy),angle,1)
    img_rotated = cv2.warpAffine(img,M,(cols,rows))
    img_rz = cv2.resize(img_rotated, (  int(ratio *cols ) , int (ratio *rows)))
    cnt = get_contour_large(img_rz)
    x,y,w,h = cv2.boundingRect(cnt)
    img_money = img_rz[y:y+h, x:x+w]
    return img_money

def matching_money(img):

    val = ""
    min_count = 500*500
    img_det_blur = cv2.blur(img, (5,5))
    for i , im in enumerate(money_template):
        h,w = im.shape
        im_rz = cv2.resize(img_det_blur, (w,h))
        img_ref_blur = cv2.blur(im, (5,5))

        eq_ref = cv2.equalizeHist(img_ref_blur)
        eq_def = cv2.equalizeHist(im_rz)
        sub_img = cv2.subtract(eq_ref, eq_def)
        _, img_threshold = cv2.threshold(sub_img, 0, 255, cv2.THRESH_OTSU)
        count = cv2.countNonZero(img_threshold)
        if(count  < min_count):
            min_count = count
            val = money_value[i]
            
    rows, cols = img_det_blur.shape
    M = cv2.getRotationMatrix2D((int(cols/2),int(rows/2)),180,1)
    img_det_blur = cv2.warpAffine(img_det_blur,M,(cols,rows))
    for i , im in enumerate(money_template):
        h,w = im.shape
        im_rz = cv2.resize(img_det_blur, (w,h))
        img_ref_blur = cv2.blur(im, (5,5))

        eq_ref = cv2.equalizeHist(img_ref_blur)
        eq_def = cv2.equalizeHist(im_rz)
        sub_img = cv2.subtract(eq_ref, eq_def)
        _, img_threshold = cv2.threshold(sub_img, 0, 255, cv2.THRESH_OTSU)
        count = cv2.countNonZero(img_threshold)
        if(count  < min_count):
            min_count = count
            val = money_value[i]
    return val
def main():
    load_template()
    files = glob.glob("input_image\\*.JPG")
    for i, item in enumerate(files):
        img = cv2.imread(item, 0)
        img_detected = money_detector(img)
        cv2.imwrite("out\\" + str(i) + ".png", img_detected)
        val = matching_money(img_detected)
        
        print(item + " is " + val )
        
if __name__ == "__main__":
    main()