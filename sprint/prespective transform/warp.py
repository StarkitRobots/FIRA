import cv2 
#import matplotlib.pylab as plt
import numpy as np

def getCalibMatrix(nx, ny, img):
    ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)
    if ret == True:
        offset = 280 # offset for dst points
        # Grab the image shape
        img_size = (img.shape[0], img.shape[1])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes

        dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                                     [img_size[0]-offset, img_size[1]-offset],
                                     [offset, img_size[1]-offset]])

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        return M
    else:
        return -1
    
def getNxNy(filenames):
    for filename in filenames:    
        for nx in range(3,8):
            for ny in range(3,8):
                img = cv2.imread(filename, 0)
                ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)
                if ret == True:
                    print(filename,nx,ny)

                    
def warpTransform(img, M):
    img_size = (img.shape[0], img.shape[1])
    warped = cv2.warpPerspective(img, M, img_size)
    return warped