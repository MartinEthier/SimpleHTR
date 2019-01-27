import numpy as np
import cv2

from line_segmentation import segment_lines

from DataLoader import Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess

class FilePaths:
	"filenames and paths to data"
	fnCharList = '../model/charList.txt'
	fnAccuracy = '../model/accuracy.txt'
	fnTrain = '../data/'
	fnInfer = '../data/test.png'
	fnCorpus = '../data/corpus.txt'

def infer(model, img):
    batch = Batch(None, [img])
    tup = model.inferBatch(batch, True)
    
    return tup

def word_conv(img, reuse):
    print(open(FilePaths.fnAccuracy).read())
    decoderType = DecoderType.BestPath
    model = Model(open(FilePaths.fnCharList).read(), reuse, decoderType, mustRestore=True)
    
    imgSize = (128, 32)
    input_img = preprocess(img, imgSize)
    rec_prob = infer(model, input_img)
    
    return rec_prob
                        
"""
def note_to_text(gray_img):
    # takes as input a binary opencv image
    
    # Segment image into separate lines
    lines = segment_lines(bw_img)
    
    
    print(open(FilePaths.fnAccuracy).read())
    decoderType = DecoderType.BestPath
    model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True)
    
    texts = []
    for line in lines:
        input_img = preprocess(line.img, (128, 32))
        
        recognized, probability = infer(model, input_img)
        line.text = recognized
        line.prob = probability
        texts.append(line.text)
    
    return texts
    
"""