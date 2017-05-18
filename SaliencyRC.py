import segmentImage
import cv2
import numpy as np
from utils import sqr,sqrDist,dist
import copy
class Region:
    def __init__(self,pixNum=0,ad2c=(0,0)):
        self.pixNum = pixNum
        self.ad2c = [ad2c[0],ad2c[1]]
        self.freIdx = []
        self.centroid = [0,0]

def GetHC(img3f):
    binN,idx1i,binColor3f,colorNums1i = Quantize(img3f)
    print(binN)
    cv2.cvtColor(binColor3f,cv2.COLOR_BGR2Lab,binColor3f)
    weight1f = np.zeros(colorNums1i.shape,np.float32)
    cv2.normalize(colorNums1i.astype(np.float32),weight1f,1,0,cv2.NORM_L1)
    colorSal = np.zeros((1,binN),np.float64)
    similar = [[] for _ in range(binN)]
    for i in range(binN):
        similar[i].append([0.0,i])
        for j in range(binN):
            if i != j:
                dij = dist(binColor3f[0,i],binColor3f[0,j])
                similar[i].append([dij,j])
                colorSal[0,i] += weight1f[0,j] * dij
        similar[i].sort()
    SmoothBySaliency(np.ones(colorSal.shape,np.int32),colorSal,0.25,similar)
    salHC1f = np.zeros((img3f.shape[0],img3f.shape[1]),np.float64)
    width = img3f.shape[1]
    height = img3f.shape[0]
    h_range = range(height)
    w_range = range(width)
    for y in h_range:
        for x in w_range:
            salHC1f[y,x] = colorSal[0,idx1i[y,x]]
    cv2.GaussianBlur(salHC1f,(3,3),0,salHC1f)
    cv2.normalize(salHC1f,salHC1f,0,1,cv2.NORM_MINMAX)
    return salHC1f

def GetFT(img3f):
    sal = np.zeros((img3f.shape[0],img3f.shape[1]),np.float32)
    tImg = cv2.GaussianBlur(img3f,(3,3),0)
    colorM = cv2.mean(tImg)
    height = img3f.shape[0]
    width = img3f.shape[1]
    for y in range(height):
        for x in range(width):
            sal[y,x] = float(sqr(colorM[0] - tImg[y,x,0]) + sqr(colorM[1] - tImg[y,x,1])+sqr(colorM[2] - tImg[y,x,2]))
    cv2.normalize(sal,sal,0,1,cv2.NORM_MINMAX)
    return sal
def GetRC(img3f,sigmaDist=0.4,segK=200,segMinSize=50,segSigma=0.5):
    imgLab3f = img3f.copy()
    cv2.cvtColor(img3f,cv2.COLOR_BGR2Lab,imgLab3f)
    regNum,regIdx1i = segmentImage.SegmentImage(imgLab3f,None,segSigma,segK,segMinSize)
    Quatizenum,colorIdx1i,color3fv,tmp = Quantize(img3f)
    if Quatizenum == 2:
        sal = colorIdx1i.copy()
        cv2.compare(colorIdx1i,1,cv2.CMP_EQ,sal)
        sal = sal.astype(np.float32)
        mn = np.min(sal)
        mx = np.max(sal)
        sal = (sal-mn)*255/(mx-mn)
    if Quatizenum <= 2:
        return np.zeros(img3f.shape,img3f.dtype)
    cv2.cvtColor(color3fv,cv2.COLOR_BGR2Lab,color3fv)
    regs = BuildRegions(regIdx1i,colorIdx1i,color3fv.shape[1],regNum)
    regSal1v = RegionContrast(regs,color3fv,sigmaDist)
    sal1f = np.zeros((img3f.shape[0],img3f.shape[1]),img3f.dtype)
    cv2.normalize(regSal1v,regSal1v,0,1,cv2.NORM_MINMAX)
    #regSal1v = regSal1v / regSal1v.max(0)
    width = img3f.shape[1]
    height = img3f.shape[0]
    height_range = range(height)
    width_range = range(width)
    for y in height_range:
        for x in width_range:
            sal1f[y,x] = regSal1v[0,regIdx1i[y,x]]
    bdgReg1u = GetBorderReg(regIdx1i,regNum,0.02,0.4)
    idxs = np.where(bdgReg1u == 255)
    sal1f[idxs] = 0
    np.set_printoptions(threshold=np.nan)
    #cv2.imwrite('rc.png',sal1f)
    SmoothByHist(img3f,sal1f,0.1)
    SmoothByRegion(sal1f,regIdx1i,regNum)
    sal1f[idxs] = 0
    cv2.GaussianBlur(sal1f,(3,3),0,sal1f)
    return sal1f

def Quantize(img3f,ratio=0.95,colorNums=(12,12,12)):
    clrTmp = [i - 0.0001 for i in colorNums]
    w = [colorNums[1] * colorNums[2],colorNums[2],1]
    idx1i = np.zeros((img3f.shape[0],img3f.shape[1]),np.int32)
    width = img3f.shape[1]
    height = img3f.shape[0]
    height_range =range(height)
    width_range = range(width)
    #build color pallet
    pallet = {}
    for y in height_range:
        for x in width_range:
            idx1i[y,x] = int(img3f[y,x,0]*clrTmp[0]) * w[0] + int(img3f[y,x,1] * clrTmp[1]) * w[1] + int(img3f[y,x,2] * clrTmp[2] )
            if idx1i[y,x] not in pallet.keys():
                pallet[idx1i[y,x]] = 1
            else:
                pallet[idx1i[y,x]] += 1
    #Find significant colors
    maxNum = 0
    num = [(pallet[key],key) for key in pallet] #(num,color) pairs in num
    num.sort(reverse=True)
    maxNum = len(num)
    maxDropNum = int(np.round(height*width*(1-ratio)))
    crnt = num[maxNum-1][0]
    while crnt < maxDropNum and maxNum > 1:
        crnt += num[maxNum-2][0]
        maxNum -= 1
    maxNum = 256 if maxNum > 256 else maxNum # To avoid very rarely case
    if maxNum <= 10:
        maxNum = 10 if len(num) > 10 else len(num)
    pallet.clear()
    for i in range(maxNum):
        pallet[num[i][1]] = i
    color3i = [[int(num[i][1] / w[0]),int(num[i][1] % w[0] / w[1]),int(num[i][1] % w[1])] for i in range(len(num))]
    for i in range(maxNum,len(num)):
        simIdx = 0
        simVal = (1 << 31) - 1 # int32 max
        for j in range(maxNum):
            d_ij = sqrDist(color3i[i],color3i[j])
            if d_ij < simVal:
                simVal = d_ij
                simIdx = j
        pallet[num[i][1]] = pallet[num[simIdx][1]]
    color3f = np.zeros((1,maxNum,3),np.float32)
    colorNum = np.zeros((1,maxNum),np.int32)
    for y in height_range:
        for x in width_range:
            idx1i[y,x] = pallet[idx1i[y,x]]
            color3f[0,idx1i[y,x]] += img3f[y,x]
            colorNum[0,idx1i[y,x]] += 1
    for i in range(color3f.shape[1]):
        color3f[0,i] /= colorNum[0,i]
    return color3f.shape[1],idx1i,color3f,colorNum

def BuildRegions(regIdx1i,colorIdx1i,colorNum,regNum):
    width = regIdx1i.shape[1]
    height = regIdx1i.shape[0]
    cx = width / 2.0
    cy = height / 2.0
    width_range = range(width)
    height_range = range(height)
    regColorFre1i = np.zeros((regNum,colorNum),np.int32)
    regs = [Region() for _ in range(regNum)]
    for y in height_range:
        for x in width_range:
            regidx = regIdx1i[y,x]
            coloridx = colorIdx1i[y,x]
            reg = regs[regidx]
            reg.pixNum += 1
            reg.centroid[0] += x # region center x coordinate
            reg.centroid[1] += y # region center y coordinate
            regColorFre1i[regidx,coloridx] += 1
            reg.ad2c[0] = abs(x - cx)
            reg.ad2c[1] = abs(y - cy)
    for i in range(regNum):
        reg = regs[i]
        reg.centroid[0] /= reg.pixNum * width
        reg.centroid[1] /= reg.pixNum * height
        reg.ad2c[0] /= reg.pixNum * width
        reg.ad2c[1] /= reg.pixNum * height
        for j in range(colorNum):
            fre = float(regColorFre1i[i,j]) / reg.pixNum
            EPS = 1e-200
            if regColorFre1i[i,j] > EPS:
                reg.freIdx.append((fre,j))
    return regs

def RegionContrast(regs,color3fv,sigmaDist):
    cDistCache1f = np.zeros((color3fv.shape[1],color3fv.shape[1]),np.float64)
    for i in range(cDistCache1f.shape[0]):
        for j in range(i+1,cDistCache1f.shape[1]):
            cDistCache1f[i,j] = dist(color3fv[0, i], color3fv[0, j])
            cDistCache1f[j,i] = cDistCache1f[i,j]
    regNum = len(regs)
    rDistCache1d = np.zeros((regNum,regNum),np.float64)
    regSal1d = np.zeros((1,regNum),np.float64)
    for i in range(regNum):
        for j in range(regNum):
            if i < j:
                dd = 0.0
                range_m = range(len(regs[i].freIdx))
                range_n = range(len(regs[j].freIdx))
                c1 = regs[i].freIdx
                c2 = regs[j].freIdx
                for m in range_m:
                    for n in range_n:
                        dd += cDistCache1f[c1[m][1],c2[n][1]] * c1[m][0] * c2[n][0]
                        #dd += c1[m][0] * c2[n][0]
                #print(dd)
                tmp = dd * np.exp(-1.0 * sqrDist(regs[i].centroid,regs[j].centroid)/sigmaDist)
                #print(tmp)
                rDistCache1d[i][j] = tmp
                rDistCache1d[j][i] = tmp
            regSal1d[0,i] += regs[j].pixNum * rDistCache1d[i,j]
        regSal1d[0,i] *= np.exp(-9.0 * (sqr(regs[i].ad2c[0])+sqr(regs[i].ad2c[1])))
    return regSal1d

def GetBorderReg(idx1i,regNum,ratio=0.02,thr=0.3):
    EPS = 1e-200
    #variance of x and y
    vX = [0.0 for i in range(regNum)]
    vY = copy.deepcopy(vX)
    # mean value of x and y, pixel number of region
    mX = copy.deepcopy(vX)
    mY = copy.deepcopy(vX)
    n = copy.deepcopy(vX)
    w = idx1i.shape[1]
    h = idx1i.shape[0]
    h_range = range(h)
    w_range = range(w)

    for y in h_range:
        for x in w_range:
            mX[idx1i[y,x]] += x
            mY[idx1i[y,x]] += y
            n[idx1i[y,x]] += 1
    for i in range(regNum):
        mX[i] /= n[i]
        mY[i] /= n[i]
    for y in h_range:
        for x in w_range:
            idx = idx1i[y,x]
            vX[idx] += abs(x - mX[idx])
            vY[idx] += abs(y - mY[idx])
    for i in range(regNum):
        vX[i] = vX[i] / n[i] + EPS
        vY[i] = vY[i] / n[i] + EPS
    # Number of border pixels in x and y border region
    xbNum = [0 for i in range(regNum)]
    ybNum = copy.deepcopy(xbNum)
    wGap = np.round(w * ratio)
    hGap = np.round(h * ratio)
    bPnts = []
    pnt = [0,0]
    sx = pnt[0]
    sy = pnt[1]
    #top region
    while pnt[1] != hGap:
        pnt[0] = sx
        while pnt[0] != w:
            ybNum[idx1i[pnt[1],pnt[0]]] += 1
            bPnts.append(copy.deepcopy(pnt))
            pnt[0] += 1
        pnt[1] +=1
    pnt = [0,int(h-hGap)]
    sx = pnt[0]
    sy = pnt[1]
    #Bottom region
    while pnt[1] != h:
        pnt[0] = sx
        while pnt[0] != w:
            ybNum[idx1i[pnt[1],pnt[0]]] += 1
            bPnts.append(copy.deepcopy(pnt))
            pnt[0] += 1
        pnt[1] += 1
    pnt = [0,0]
    sx = pnt[0]
    sy = pnt[1]
    #Left Region
    while pnt[1] != h:
        pnt[0] = sx
        while pnt[0] != wGap:
            ybNum[idx1i[pnt[1],pnt[0]]] += 1
            bPnts.append(copy.deepcopy(pnt))
            pnt[0] += 1
        pnt[1] += 1
    pnt = [int(w - wGap),0]
    sx = pnt[0]
    sy = pnt[1]
    #Right Region
    while pnt[1] != h:
        pnt[0] = sx
        while pnt[0] != w:
            ybNum[idx1i[pnt[1],pnt[0]]] += 1
            bPnts.append(copy.deepcopy(pnt))
            pnt[0] += 1
        pnt[1] += 1
    bReg1u = np.zeros((h,w),np.uint8)
    #likelihood map of border region
    xR = 1.0 / (4 * hGap)
    yR = 1.0 / (4 * wGap)
    regL = [0 for _ in range(regNum)]
    for i in range(regNum):
        lk = xbNum[i] * xR / vY[i] + ybNum[i] * yR / vX[i]
        regL[i] = 255 if lk / thr > 1 else 0
    for y in h_range:
        for x in w_range:
            bReg1u[y,x] = regL[idx1i[y,x]]
    length = len(bPnts)
    len_range = range(length)
    for i in len_range:
        bReg1u[bPnts[i][1],bPnts[i][0]] = 255
    return bReg1u

def SmoothByHist(img3f,sal1f,delta):
    #Quantize colors
    binN, idx1i, binColor3f, colorNums1i = Quantize(img3f)
    _colorSal = np.zeros((1,binN),np.float64)
    height = img3f.shape[0]
    width = img3f.shape[1]
    h_range = range(height)
    w_range = range(width)
    for y in h_range:
        for x in w_range:
            _colorSal[0,idx1i[y,x]] += sal1f[y,x]
    for i in range(binN):
        _colorSal[0,i] /= colorNums1i[0,i]
    cv2.normalize(_colorSal,_colorSal,0,1,cv2.NORM_MINMAX)
    #Find similar colors & Smooth saliency value for color bins
    similar = [[] for i in range(binN)]
    cv2.cvtColor(binColor3f,cv2.COLOR_BGR2Lab,binColor3f)
    for i in range(binN):
        similar[i].append([0.0,i])
        for j in range(binN):
            if i != j:
                similar[i].append([dist(binColor3f[0,i],binColor3f[0,j]),j])
        similar[i].sort()

    cv2.cvtColor(binColor3f,cv2.COLOR_Lab2BGR,binColor3f)
    SmoothBySaliency(colorNums1i,_colorSal,delta,similar)
    #reassign pixel saliency values
    for y in h_range:
        for x in w_range:
            sal1f[y,x] = _colorSal[0,idx1i[y,x]]

def SmoothByRegion(sal1f,idx1i,regNum,bNormalize = True):
    saliecy = [0.0 for _ in range(regNum)]
    counter = [0 for _ in range(regNum)]
    h = sal1f.shape[0]
    w = sal1f.shape[1]
    h_range = range(h)
    w_range = range(w)
    for y in h_range:
        for x in w_range:
            saliecy[idx1i[y,x]] += sal1f[y,x]
            counter[idx1i[y,x]] += 1
    for i in range(len(counter)):
        saliecy[i] /= counter[i]

    rSal = np.array([saliecy],dtype=np.float64)
    if(bNormalize):
        cv2.normalize(rSal,rSal,0,1,cv2.NORM_MINMAX)
    for y in h_range:
        for x in w_range:
            sal1f[y,x] = saliecy[idx1i[y,x]]

def SmoothBySaliency(colorNums1i,sal1f,delta,similar):
    if sal1f.shape[1] < 2:
        return
    binN = sal1f.shape[1]
    newSal1d = np.zeros((1,binN),np.float64)
    tmpNum = int(np.round(binN*delta))
    n =  tmpNum if tmpNum > 2 else 2
    dist = [0.0 for _ in range(n)]
    val = copy.deepcopy(dist)
    w = copy.deepcopy(dist)
    binN_range = range(binN)
    n_range = range(n)
    for i in binN_range:
        totalDist = 0.0
        totalWeight = 0.0
        for j in n_range:
            ithIdx = similar[i][j][1]
            dist[j] = similar[i][j][0]
            val[j] = sal1f[0,ithIdx]
            w[j] = colorNums1i[0,ithIdx]
            totalDist += dist[j]
            totalWeight += w[j]
        valCrnt = 0.0
        for j in n_range:
            valCrnt += val[j] * (totalDist - dist[j]) * w[j]
        #print(valCrnt,totalDist,totalWeight)
        newSal1d[0,i] = valCrnt / (totalDist * totalWeight)
    cv2.normalize(newSal1d,sal1f,0,1,cv2.NORM_MINMAX)







