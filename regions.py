import cv2 as cv
import fitz
from matplotlib import pyplot as plt
import numpy as np
ekg = 'ekg.pdf'
doc = fitz.open(ekg)
page = doc.load_page(0)
page.set_rotation(270)
pix = page.get_pixmap()
output = 'outfile.jpeg'
pix.pil_save(output)


im = cv.imread(output)
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.treshhold(imgray, 15, 255, cv2.THRESH_BINARY_INV)
image, contours, hierarchy = cv.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
frame = cv.drawContours(frame, contours, -1,(0,0,255),3)
plt.imshow(frame)		
#plt.imshow(mask)
plt.show()
#plt.imshow(imgray, cmap = 'gray')
#plt.show()