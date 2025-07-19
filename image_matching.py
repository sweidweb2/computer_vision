import cv2 as cv
import imutils
import numpy as np
import matplotlib.pyplot as plt

def d(image):
    cv.imshow("new",image)
    cv.waitKey(0)

def allign_images(image,template,maxFeatures=500,keepPercent=0.2,debug=False):
    imagegray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    templategray=cv.cvtColor(template,cv.COLOR_BGR2GRAY)

    orb=cv.ORB_create(maxFeatures)
    (kpsA,descA)=orb.detectAndCompute(imagegray,None)
    (kpsB, descB) = orb.detectAndCompute(templategray,None)

    method=cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher=cv.DescriptorMatcher_create(method)
    matches=matcher.match(descA,descB)

    matches=sorted(matches,key=lambda x:x.distance)
    keep=int(len(matches)*keepPercent)
    matches=matches[:keep]

    ptsA=np.zeros([len(matches),2],dtype='float')
    ptsB=np.zeros([len(matches),2],dtype='float')

    if debug:
        matchedvis=cv.drawMatches(image,kpsA,template,kpsB,matches,None)
        matchedvis=imutils.resize(matchedvis,width=1300)
        cv.imshow("matched keypoints",matchedvis)
        cv.waitKey(0)

    for(i,m) in enumerate(matches):
        ptsA[i]=kpsA[m.queryIdx].pt
        ptsB[i]=kpsB[m.trainIdx].pt

    (H,mask)=cv.findHomography(ptsA,ptsB,method=cv.RANSAC)
    (h,w)=template.shape[:2]

    alligned=cv.warpPerspective(image,H,(w,h))
    return alligned


img=cv.imread('Lab2/image.jpg')
# d(img)

main=cv.imread('Lab2/main.png')
# d(main)


alligneed=allign_images(img,main,debug=True)

d(alligneed)






