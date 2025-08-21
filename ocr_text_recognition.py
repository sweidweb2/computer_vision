
import cv2 as cv
import matplotlib.pyplot as plt
import easyocr


# read image
img=cv.imread("data/test3.png")



#instance text editor
reader=easyocr.Reader(['en'],gpu=False)


# detect text and image
text=reader.readtext(img)

# for t in text:
#     print(t)
# print(text)
threshold=0.25
# draw bbox and text
for bbox, text1, prob in text:

    bbox = [[int(x), int(y)] for [x, y] in bbox]
    print(bbox, text1, float(prob))
    if prob>threshold:
        cv.rectangle(img,bbox[0],bbox[2],(0,255,0),5)
        cv.putText(img,text1,bbox[0],cv.FONT_HERSHEY_COMPLEX,0.65,(255,0,0),1)

plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
plt.show()