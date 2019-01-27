import numpy as np
import cv2
from matplotlib import pyplot as plt
from text_convert import word_conv
from line_segmentation import segment_lines
import tensorflow as tf

newpath = "C:\\Users\\Martin\\Desktop\\DeltaHacks V\\SimpleHTR\\data\\test_images\\Database_Changes.png"

img = cv2.imread(newpath, cv2.IMREAD_GRAYSCALE)
#print(img.shape)

lines = segment_lines(img)

#print(len(lines))

#plt.imshow(lines[0].img)
#plt.show()

#plt.imshow(lines[1].img)
#plt.show()

for line in lines:
    text_prob = word_conv(line.img, tf.AUTO_REUSE)
    line.text = text_prob[0]
    line.prob = text_prob[1]
    tf.reset_default_graph()
    print(text_prob)

