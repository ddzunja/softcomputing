import os
import sys
import time
import numpy as np
import cv2
import itertools
from scipy import ndimage
from vector import distance, pnt2line
from keras.models import load_model
from trainClassifier import trainNM, getAreaOfIntrest

sampleArr = ['video-0.avi', 'video-1.avi', 'video-2.avi',
             'video-3.avi', 'video-4.avi', 'video-5.avi', 
             'video-6.avi', 'video-7.avi', 'video-8.avi', 
             'video-9.avi']
 ############ FUNKCIJE ##############

cc = -1
def nextId():
    global cc
    cc += 1
    return cc

def inRange(r, item, items):
    retVal = []
    for obj in items:
        mdist = distance(item['center'], obj['center'])
        if(mdist<r):
            retVal.append(obj)
    return retVal


elements = []
t = 0
counter = 0
addition = 0
subtract = 0
times = []
subArray = []
addArray = []

def deskew(img):
    m = cv2.moments(img)

    SZ = 28

    if abs(m['mu02']) < 1e-2:
        # no deskewing needed. 
        return img
    # Calculate skew based on central momemts. 
    skew = m['mu11']/m['mu02']
    # Calculate affine transform to correct skewness. 
    #M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    M = np.array([[1, skew, -0.5*SZ*skew], [0, 1, 0]], 'float32')
    # Apply affine transform
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def classify(img, point, classifier):

    x, y = point

    global counter

    cropImage = img[y-12:y+12, x-12:x+12]

    cropImage = cv2.cvtColor(cropImage, cv2.COLOR_BGR2GRAY)

    cropImage = deskew(cropImage)
    

    cropImage = cv2.dilate(cropImage, (4, 4))

    cropImage = getAreaOfIntrest(cropImage)

    toPredict = cropImage.flatten() / 255.0

    toPredict = (np.array([toPredict], 'float32'))

    classifiedNumber = np.argmax(classifier.predict(toPredict)[0])

    #cv2.imshow('classifier', cropImage)

    return classifiedNumber


def trackObjects(img, linesFinal, classifier):
    start_time = time.time()
    origImg = img.copy()
    #(lower, upper) = boundaries[0]
    # create NumPy arrays from the boundaries

    global elements
    global counter
    global times
    global subtract
    global addition
    global t

    kernel = np.ones((2, 2),np.uint8)
    lower = np.array([230, 230, 230])
    upper = np.array([255, 255, 255])

    lineAdd = linesFinal['add']
    lineSub = linesFinal['sub']
    

    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    mask = cv2.inRange(img, lower, upper)    
    img0 = 1.0*mask

    img0 = cv2.dilate(img0,kernel)
    img0 = cv2.dilate(img0,kernel)

    #cv2.imshow('proba', img0)
    labeled, nr_objects = ndimage.label(img0)
    objects = ndimage.find_objects(labeled)
    for i in range(nr_objects):
        loc = objects[i]
        
        (xc,yc) = ((loc[1].stop + loc[1].start)/2,
                   (loc[0].stop + loc[0].start)/2)
        (dxc,dyc) = ((loc[1].stop - loc[1].start),
                   (loc[0].stop - loc[0].start))

        if(dxc>10 or dyc>10):

            xc = int(xc)
            yc = int(yc)
            dxc = int(dxc)
            dyc = int(dyc)
            cv2.circle(img, (xc,yc), 16, (25, 25, 255), 1)
            elem = {'center':(xc,yc), 'size':(dxc,dyc), 't':t}
            # find in range
            lst = inRange(20, elem, elements)
            nn = len(lst)
            if nn == 0:
                elem['id'] = nextId()
                elem['t'] = t
                elem['passAdd'] = False
                elem['passSub'] = False
                elem['history'] = [{'center':(xc,yc), 'size':(dxc,dyc), 't':t}]
                #elem['number'] = classify(origImg, (xc, yc), classifier)
                elem['number'] = None
                elem['future'] = [] 
                elements.append(elem)
            elif nn == 1:
                lst[0]['center'] = elem['center']
                lst[0]['t'] = t
                lst[0]['history'].append({'center':(xc,yc), 'size':(dxc,dyc), 't':t}) 
                lst[0]['future'] = [] 
                        
    for el in elements:
        tt = t - el['t']
        if(tt<3):
            ####
            if el['number'] is None:
                el['number'] = classify(origImg, el['center'], classifier)

            dist, pnt, r = pnt2line(el['center'], lineAdd[0], lineAdd[1])
            c = None
            passed = False
            if r>0:
                passed = True
                cv2.line(img, pnt, el['center'], (0, 255, 25), 1)
                c = (25, 25, 255)
                if(dist<9):
                    c = (0, 255, 160)
                    if el['passAdd'] == False:
                        el['passAdd'] = True
                        counter += 1
                        addition += el['number']
                        addArray.append(el['number'])

                #cv2.circle(img, el['center'], 16, c, 2)

            dist, pnt, r = pnt2line(el['center'], lineSub[0], lineSub[1])
            if r>0:
                passed = True
                cv2.line(img, pnt, el['center'], (255, 25, 0), 1)
                c = (25, 25, 255)
                if(dist<9):
                    c = (0, 255, 160)
                    if el['passSub'] == False:
                        el['passSub'] = True
                        counter += 1
                        subtract -= el['number']
                        subArray.append(el['number'])

            if passed:
                cv2.circle(img, el['center'], 16, c, 2)
            
            

            id = el['id']
            #####
            if el['number'] is not None:
                cv2.putText(img, text = str(el['number']), 
                    org = (el['center'][0]+15, el['center'][1]+20), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = (0, 0, 255))
            for hist in el['history']:
                ttt = t-hist['t']
                if(ttt<100):
                    cv2.circle(img, hist['center'], 1, (0, 255, 255), 1)

            for fu in el['future']:
                ttt = fu[0]-t
                if(ttt<100):
                    cv2.circle(img, (fu[1], fu[2]), 1, (255, 255, 0), 1)

    elapsed_time = time.time() - start_time
    times.append(elapsed_time*1000)
    cv2.putText(img, text = 'Add: ' + str(addition), org = (480, 40), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (90,90,255))
    cv2.putText(img, text = 'Sub: ' + str(subtract), org = (480, 60), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (90,90,255))    
    cv2.putText(img, text = 'Sum: ' + str(addition + subtract), org = (480, 80), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (90,90,255))    
    cv2.putText(img, text = 'Counter: ' + str(counter), org = (480, 100), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (90,90,255))    

    t += 1


def detectLines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 10, 100, apertureSize = 3)

    linesTemp = cv2.HoughLinesP(edges, 2, np.pi/180, 100, maxLineGap = 20, minLineLength = 150)

    lines = []
    middlePoints = []

    for i in range(0, len(linesTemp)):
        x1, y1, x2, y2 = linesTemp[i][0]
        lines.insert(len(lines), linesTemp[i][0])
        middlePoints.insert(len(middlePoints), ((x2 + x1) / 2, (y2 + y1) / 2, i))
        

    ## Trazimo dve linije cija je udaljenost sredisnjih tacaka najveca

    longestDist = 0
    firstLine = 0
    secondLine = 0

    for i in range(0, len(middlePoints)):
        for j in range(i, len(middlePoints)):
            x1, y1, pointNum1 = middlePoints[i]
            x2, y2, pointNum2 = middlePoints[j]
            longDistTemp = ((x2 - x1) ** 2 + (y2 - y2) ** 2) ** (1/2.0)
            if longDistTemp > longestDist:
                firstLine = pointNum1
                secondLine = pointNum2
                longestDist = longDistTemp

    
    linesFinal = {}

    x1, y1, x2, y2 = lines[firstLine]
    firstMidPoint = ((x2 + x1) / 2, (y2 + y1) / 2)
    linesFinal['add'] = ((x1, y1), (x2, y2))
    x1, y1, x2, y2 = lines[secondLine]
    secondMidPoint = ((x2 + x1) / 2, (y2 + y1) / 2)
    linesFinal['sub'] = ((x1, y1), (x2, y2))

    if distance((20, 450), firstMidPoint) < distance((20, 450), secondMidPoint):
        linesFinal['add'], linesFinal['sub'] = linesFinal['sub'], linesFinal['add']

    
    return linesFinal
     
def main(vidTitle, classifier, show):

    global elements
    global t
    global counter
    global addition
    global subtract
    global times
    global subArray
    global addArray
   

    print('\n----------------')

    if show is None:
        show = True

    cap = cv2.VideoCapture(vidTitle)

    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameCounter = 0

    lines = None

    while(cap.isOpened()):
        ret, frame = cap.read()

        prnStr = '\rAnalyzing ' + vidTitle + ': ' + str(round((frameCounter / (frameCount * 1.0) * 100.0), 1)) + '%, '
        prnStr += 'Added = ' + str(addArray) + ', '
        prnStr += 'Subtracted = ' + str(subArray)

        print(prnStr, end='')
  
        frameCounter += 1

        if ret is True:

            if lines is None:
                lines = detectLines(frame)

            addLine = lines['add']
            subLine = lines['sub']

            trackObjects(frame, lines, classifier)
              
            #cv2.line(frame, addLine[0], addLine[1], (255, 170, 255), 2)
            #cv2.line(frame, subLine[0], subLine[1], (255, 0, 0), 2)
            cv2.putText(frame, text = 'ADD + ', org = addLine[0], fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (90, 255, 255))
            cv2.putText(frame, text = 'SUB - ', org = subLine[0], fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (90, 255, 255)) 
            
            if show is True:
                cv2.imshow(vidTitle, frame)

            # kraj
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()

    cv2.destroyAllWindows()


    sum = addition + subtract

    
    print('\nResult = ' + str(sum) +', Addition = ' + str(addition) + ', Subtraction = ' + str(subtract) + ', Counter = ' + str(counter))
    print('Finished ' + vidTitle)
    print('----------------\n')
    
    elements = []
    t = 0
    counter = 0
    addition = 0
    subtract = 0
    times = []
    subArray = []
    addArray = []

    return sum



###### INIT ######
os.system('cls')

resultArray = []

if os.path.exists('model.h5') is False:
    print('model.h5 file not found! Training...')
    trainNM()

model = load_model('model.h5')

if sys.argv[1] == 'all':

    for i in range(0, 10):
        result = main(vidTitle = sampleArr[i], classifier = model, show = False)
        resultArray.append(result)

    f = open('out.txt', 'w')

    strWr = 'RA 52/2014 Dejan Dzunja\n'
    strWr += 'file\tsum\n'

    for i in range(0, 10):
        strWr += sampleArr[i] + ' ' + str(resultArray[i]) + '\n'

    f.write(strWr)
    f.close()

    print('-----------')
    print(strWr)
    print('-----------')

    import test



elif int(sys.argv[1]) >= 0 and int(sys.argv[1]) <= 9:
    main(vidTitle = sampleArr[int(sys.argv[1])], classifier = model, show = True)
