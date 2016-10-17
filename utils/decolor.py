import cv2
import sys

img_name = sys.argv[1]

img = cv2.imread(sys.argv[1])
original = img.copy()
img = cv2.resize(img, (160,144), interpolation=cv2.INTER_CUBIC)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('images/gray.png', img)

original = cv2.resize(original, (160*2,144*2), interpolation=cv2.INTER_CUBIC)
cv2.imshow('image',original)
cv2.waitKey(0)
cv2.destroyAllWindows()

