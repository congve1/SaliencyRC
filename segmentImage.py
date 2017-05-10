import segmentGraph
import cv2
import numpy as np
from math import sqrt,pow
# dissimilarity measure between pixels
def diff(img3f,x1,y1,x2,y2):
    p1 = img3f[y1,x1]
    p2 = img3f[y2,x2]
    return np.sqrt(np.sum(np.power(p1-p2,2)))

def SegmentImage(src3f,imgInd,sigma=0.5,c=200,min_size=50):
    width = src3f.shape[1]
    height = src3f.shape[0]
    smImg3f = np.zeros(src3f.shape,dtype=src3f.dtype)
    cv2.GaussianBlur(src3f,(0,0),sigma,dst=smImg3f,borderType=cv2.BORDER_REPLICATE)
    # build graph
    edges = [segmentGraph.Edge() for _ in range(width * height * 4)]
    num = 0
    width_range = range(width)
    height_range = range(height)
    for y in height_range:
        for x in width_range:
            if x < width - 1:
                edges[num].a = y * width + x
                edges[num].b = y * width + (x+1)
                edges[num].w = diff(smImg3f,x,y,x+1,y)
                num += 1
            if y < height - 1:
                edges[num].a = y * width + x
                edges[num].b = (y+1) * width + x
                edges[num].w = diff(smImg3f,x,y,x,y+1)
                num += 1
            if x < (width - 1) and y < (height-1):
                edges[num].a = y * width + x
                edges[num].b = (y+1)*width + (x+1)
                edges[num].w = diff(smImg3f,x,y,x+1,y+1)
                num += 1
            if x < (width - 1) and y > 0:
                edges[num].a = y * width + x
                edges[num].b = (y-1) * width + (x+1)
                edges[num].w = diff(smImg3f,x,y,x+1,y-1)
                num += 1
    # segment
    u = segmentGraph.segment_graph(width * height,num,edges,c)

    # post process small components
    num_range = range(num)
    for i in num_range:
        a = u.find(edges[i].a)
        b = u.find(edges[i].b)
        if ((a != b) and ((u.size(a) < min_size) or (u.size(b) < min_size))):
            u.join(a,b)

    # pick random colors for each components
    marker = {}
    imgInd = np.zeros(smImg3f.shape,np.int32)
    idxNum = 0
    for y in height_range:
        for x in width_range:
            comp = u.find(y * width + x)
            if comp not in marker.keys():
                marker[comp] = idxNum
                idxNum += 1
            idx = marker[comp]
            imgInd[y,x] = idx
    return idxNum,imgInd


