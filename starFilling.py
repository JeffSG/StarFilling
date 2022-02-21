import sys
import cv2
import numpy
import math

jpegLow = 128
# to calculate star color and painting, using backgroundFloor as boundry
backgroundFloor = 20
minStarSize = 3
# star radius = half lightness radius * halfFactor
halfFactor = 1.4
maxLightValue = 256
saturationFactor = 1.0
# starRange * radius is the real affected range of a star
starRange = 2

def calcRadius(img, centX, centY):
	halfLight = img[centY][centX] / 2
	for i in range(1, 9999):
		if (centX + i < width and img[centY][centX + i] <= halfLight) or (i <= centX and img[centY][centX - i] <= halfLight) or (centY + i < height and img[centY + i][centX] <= halfLight)	or (i <= centY and img[centY - i][centX] <= halfLight):
			return int(i * halfFactor + 0.5)
	return 0

def mixColor(starColor, bgColor, factor):
	c = int(starColor * factor + bgColor * (1 - factor))
	if 255 < c:
		return 255
	elif c < 0:
		return 0
	else:
		return c

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

def starFactor(rangeSize, dist):
	shrinkLightness = min(0.1 * rangeSize, 1.0)
	starListSize = 81
	saturationList = [0.000682353,0.000682353,0.000682353,0.000682353,0.000682353,0.024205882,0.024205882,0.024672619,0.079402825,0.104718094,0.15372815,0.190055902,0.266130065,0.289685714,0.390419394,0.548326625,0.604897095,0.652843529,0.696590284,0.792074484,0.824208791,0.883785919,0.918781811,0.942623399,0.948239684,0.975182151,0.975182151,0.975182151,0.975182151,0.975182151,0.975182151,0.930020778,0.930020778,0.930020778,0.924024444,0.901970871,0.901970871,0.901970871,0.8897635,0.826508508,0.788501805,0.720542467,0.704707983,0.688178069,0.667204089,0.64955117,0.620189769,0.612523392,0.588285799,0.566012085,0.563969737,0.560613636,0.536599592,0.536599592,0.533935115,0.533935115,0.533935115,0.533935115,0.531217875,0.531217875,0.531217875,0.531217875,0.531217875,0.493271557,0.493271557,0.493271557,0.480472234,0.480472234,0.476984043,0.476984043,0.476512547,0.476512547,0.476512547,0.443795927,0.443795927,0.443795927,0.438855008,0.433914089,0.43387437,0.43383465,0.433039316]
	valueList = [0.999,0.999,0.999,0.999,0.999,0.999,0.999,0.995966667,0.986866667,0.9808,0.974733333,0.950466667,0.929233333,0.8989,0.892833333,0.8716,0.8352,0.792733333,0.7442,0.710833333,0.6623,0.6077,0.565233333,0.5349,0.510633333,0.468166667,0.440866667,0.428733333,0.404466667,0.3802,0.355933333,0.3165,0.295266667,0.2892,0.283133333,0.267966667,0.264933333,0.240666667,0.222466667,0.201233333,0.195166667,0.173933333,0.158766667,0.155733333,0.1436,0.1254,0.1254,0.119333333,0.113266667,0.104166667,0.0981,0.0799,0.0799,0.0799,0.076866667,0.076866667,0.0708,0.067766667,0.058666667,0.055633333,0.055633333,0.055633333,0.055633333,0.040466667,0.037433333,0.0253,0.0253,0.022266667,0.019233333,0.019233333,0.019233333,0.019233333,0.019233333,0.013166667,0.0071,0.0071,0.0071,0.0071,0.004066667,0.001033333,0.001033333]

	pos = float(dist) / rangeSize * (starListSize - 1)
	pos = min(max(int(pos + 0.5), 0), starListSize - 1)
	return saturationList[pos], valueList[pos] * shrinkLightness


if len(sys.argv) < 7:
	exit('starFilling <centroidMask> <smallStarMask> <largeStarMask> <noStarPic> <largeStarPic> <outputPic>')

centroidFile = sys.argv[1]
smallStarMaskFile = sys.argv[2]
largeStarMaskFile = sys.argv[3]
noStarPicFile = sys.argv[4]
largeStarPicFile = sys.argv[5]
outputPicFile = sys.argv[6]

print('loading masks...')
centroidImg = cv2.imread(centroidFile, cv2.IMREAD_GRAYSCALE)
height, width = centroidImg.shape

smallStarMaskImg = cv2.imread(smallStarMaskFile, cv2.IMREAD_GRAYSCALE)
largeStarMaskImg = cv2.imread(largeStarMaskFile, cv2.IMREAD_GRAYSCALE)
print('done')

# get centroids
starCount = 0
starInfo = numpy.empty(shape = [0, 4], dtype = int)

# find star centroids and calculate the radius
print('finding stars...')
for row in range(height):
	for col in range(width):
		if jpegLow < centroidImg[row][col]:
			# identify a new star. calculate centroid.
			ptStack = numpy.empty(shape = [0, 2], dtype = int)
			ptStack = numpy.append(ptStack, [[row, col]], axis = 0)
			sumX, sumY, numPt = 0, 0, 0
			while 0 < len(ptStack):
				curX = ptStack[-1, 1]
				curY = ptStack[-1, 0]
				ptStack = ptStack[:-1, 0:]
				centroidImg[curY][curX] = 0
				sumX += curX
				sumY += curY
				numPt += 1
				if 0 < curX and jpegLow < centroidImg[curY][curX - 1]:
					ptStack = numpy.append(ptStack, [[curY, curX - 1]], axis = 0)
				if curX < width - 1 and jpegLow < centroidImg[curY][curX + 1]:
					ptStack = numpy.append(ptStack, [[curY, curX + 1]], axis = 0)
				if 0 < curY and jpegLow < centroidImg[curY - 1][curX]:
					ptStack = numpy.append(ptStack, [[curY - 1, curX]], axis = 0)
				if curY < height - 1 and jpegLow < centroidImg[curY + 1][curX]:
					ptStack = numpy.append(ptStack, [[curY + 1, curX]], axis = 0)
			if minStarSize <= numPt:
				centX = int((float(sumX)) / numPt + 0.5)
				centY = int((float(sumY)) / numPt + 0.5)
				smallRadius = calcRadius(smallStarMaskImg, centX, centY)
				largeRadius = calcRadius(largeStarMaskImg, centX, centY)
				if 0 < smallRadius and 0 < largeRadius:
					starInfo = numpy.append(starInfo, [[centY, centX, smallRadius, largeRadius]], axis = 0)
print('done')

# calculate color and paint
noStarImg = cv2.imread(noStarPicFile, cv2.IMREAD_COLOR)
largeStarImg = cv2.imread(largeStarPicFile, cv2.IMREAD_COLOR)

print('rendering...')
for star in starInfo:
	blueAcc = 0
	greenAcc = 0
	redAcc = 0
	totalWeighting = 0
	centY, centX, smallR, largeR = star[0], star[1], star[2], star[3]
	left = centX - largeR if largeR <= centX else 0
	right = centX + largeR + 1 if centX + largeR + 1 < width else width
	top = centY - largeR if largeR <= centY else 0
	bottom = centY + largeR + 1 if centY + largeR + 1 < height else height
	for row in range(top, bottom):
		for col in range(left, right):
			if largeStarMaskImg[row][col] < backgroundFloor:
				continue
			weighting = maxLightValue - largeStarMaskImg[row][col]
			blueAcc += largeStarImg[row][col][0] * weighting
			greenAcc += largeStarImg[row][col][1] * weighting
			redAcc += largeStarImg[row][col][2] * weighting
			totalWeighting += weighting
	if 0 == totalWeighting:
		print('failed on star({},{}), R={}, r={}'.format(str(centX), str(centY), str(largeR), str(smallR)))
		continue
	H, S, V = rgb2hsv(float(redAcc) / totalWeighting, float(greenAcc) / totalWeighting, float(blueAcc) / totalWeighting)
	maxS = 1.0 - (1.0 - S) / saturationFactor if 1 < saturationFactor else 0 if saturationFactor <= 0 else S * saturationFactor
	outSize = int(smallR * starRange + 0.5)
	for row in range(centY - outSize, centY + outSize + 1):
		if row < 0 or height <= row:
			continue
		for col in range(centX - outSize, centX + outSize + 1):
			if col < 0 or width <= col:
				continue
			dist = pow((row - centY) * (row - centY) + (col - centX) * (col - centX), 0.5)
			ss, sv = starFactor(outSize, dist)
			glowR, glowG, glowB = hsv2rgb(H, maxS * ss, 1.0)
			curRGB = noStarImg[row][col]
			mixR, mixG, mixB = mixColor(glowR, curRGB[2], sv ** 1.5), mixColor(glowG, curRGB[1], sv ** 1.5), mixColor(glowB, curRGB[0], sv ** 1.5)
			noStarImg[row][col] = [mixB, mixG, mixR]

print('done')

cv2.imwrite(outputPicFile, noStarImg)
