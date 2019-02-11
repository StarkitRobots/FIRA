import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
#%matplotlib inline 


#filename = "test_lines.jpg"
filename = sys.argv[1]
print(filename)
rho =1 # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180 # angular resolution in radians of the Hough grid
threshold = 100	 # minimum number of votes (intersections in Hough grid cell)
#min_line_length = 500 #minimum number of pixels making up a line
#max_line_gap = 20


def diff_neighb_1(_):
    tmp = []
    for i in range(len(_) - 1):
        tmp.append(_[i+1][0] - _[i][0])
    return tmp


def diff_neighb_2(_):
    tmp = []
    for i in range(len(_) - 1):
        tmp.append(_[i+1][1] - _[i][1])
    return tmp


def split_lines(coef, mode=1, eps = 20):
    '''
    @input =list
    @output = list of lists
    '''
    if mode == 1:
        coef.sort(key=lambda tup: tup[0])
        diff = diff_neighb_1(coef)
        #eps = max(coef, key=lambda tup: tup[0])[0] / acc

    elif mode == 2:
        coef.sort(key=lambda tup: tup[1])
        diff = diff_neighb_2(coef)
        #eps = max(coef, key=lambda tup: tup[1])[1] / acc
        
    i = 0
    tmp = []
    for el in diff:
        if el > eps:
            tmp.append(i)
        i+=1
    res = []
    prev = 0
    tmp.append(len(coef))
    for el in tmp:
        if (el-prev) < 2:
            continue
        res.append(coef[prev:el])
        prev = el + 1
    res
    return res


def mean_lines(splited_coef):
    res = []
    for el in splited_coef:
        tmp1, tmp2 = [], []
        for _ in el:
            tmp1.append(_[0])
            tmp2.append(_[1])
        res.append((np.mean(tmp1), np.mean(tmp2)))
    return res

def get_lines(filename):
    img = cv2.imread(filename)
    gray = cv2.imread(filename, 0)

    new = cv2.Canny(gray, 250, 270)
    lines = cv2.HoughLinesP(new, rho, theta, threshold)
    return lines

coef = []
for j in range(lines.shape[0]):
    for x1, y1, x2, y2 in lines[j]:
        if y1 == y2:
            a = 0
        elif x1 == x2:
            continue
        else:
            a = (y1-y2)/(x1 - x2)
        b = y1 - a*x1 
        coef.append((a,b))
        
        
splited = split_lines(coef, eps=0.3)
for el in splited:
    s_l = split_lines(el, mode=2, eps=50)
    for a, b in mean_lines(s_l):
        #print(a,b)
        if (a != a) or (np.abs(a) == 1):
            continue
        pt1 = (0, int(b))
        if a == 0:
            pt2 = (720, int(b))
        else:
            pt2 = (720, int(720*a + b))
        cv2.line(img, pt1, pt2, (0,0,0), 2)

        
cv2.imwrite("res/"+filename, img)
