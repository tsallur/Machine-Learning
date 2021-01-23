# Thomas Sallurday
# Professor Hodges
# CPSC 6430
# 4 December 2020
import numpy as np
# function to calculate weights using the normal equation
def weightCalc(Darr,y):
    return np.dot(np.linalg.pinv(np.dot(Darr.T,Darr)),np.dot(Darr.T,y))
#function to calculate J value 
def jValCalc(x,y,weights,m):
    XW = np.dot(x,weights)
    XW = np.subtract(XW,y)
    XW = np.multiply(XW,XW)
    M1m = np.ones(m)
    val = np.dot(M1m,XW)
    return np.divide(val,(2 * m))
#function to calculate R^2
def calcRSquared(jVal,yValVector,rows):
    yTotal = 0
    M1m = np.ones(rows)
    for i in range(rows):
        yTotal = yTotal + yValVector[i] 
    yMean = yTotal / rows
    newYVector = np.zeros([rows,1])
    
    for i in range(rows):
        yVal = yValVector[i] - yMean
        sqrdYVal = yVal * yVal
        newYVector[i] = sqrdYVal

    top = np.dot(M1m,newYVector)
    bottom = rows * 2
    denom =  top / bottom
    return 1 - (jVal / denom)
#function to calculate Adjusted R^2
def calcAdjustedRSquared(rSquared,rows,cols):
    top = (1 - rSquared) * (rows - 1)
    bottom = rows - cols - 1
    
    return 1 - (top / bottom)
    
str1 = input("Please enter the filename of the training set file: ")
trainData = open(str1,'r')
str1 = trainData.readline()
str1 = str1.split('\t')
rows = int(str1[0])
cols = int(str1[1])
ogData = np.zeros([rows, cols]) #stores original data
yValVector = np.zeros([rows,1]) #stores y values
for i in range(rows): # nested for loop puts data in 2d array
    str1 = trainData.readline()
    line = str1.split("\t")
    for j in range(cols + 1):
        if(j == cols):
            yValVector[i] = float(line[j])
        else:
            ogData[i][j] = float(line[j])
trainData.close() #copies all data into the 

sqrdData = np.zeros([rows, cols]) # stores squared data in 2d array
for i in range(rows): # nested for loop creates 
    for j in range(cols):
        sqrdData[i][j] = (ogData[i][j]) * (ogData[i][j])

ogAndsqrdData = np.zeros([rows, (cols * 2)]) # stores original and sqrd data
ogjCounter = 0
sqrdjCounter = 0
for i in range(rows): # puts original and squared data into the array
    ogjCounter = 0
    sqrdjCounter = 0
    for j in range((cols * 2)):
        if (j %  2 == 0):
            ogAndsqrdData[i][j] = ogData[i][ogjCounter]
            ogjCounter = ogjCounter + 1
        else:
            ogAndsqrdData[i][j] = sqrdData[i][sqrdjCounter]
            sqrdjCounter = sqrdjCounter + 1

print("\n")         
ogWeights = weightCalc(ogData,yValVector)
sqrdWeights = weightCalc(sqrdData,yValVector)
sqrdAndOgWeights = weightCalc(ogAndsqrdData,yValVector)

jValForOg = jValCalc(ogData,yValVector,ogWeights,rows)
jValForSqrd = jValCalc(sqrdData,yValVector,sqrdWeights,rows)
jValforOgAndSqrd = jValCalc(ogAndsqrdData,yValVector,sqrdAndOgWeights,rows)

Strings = []
for i in range(cols):
    Strings.append("x" + str((i + 1)) + " weight is: ")

Strings12 = []
for i in range(cols * 2):
    Strings12.append("x" + str((i + 1)) + " weight is: ")
    
print("The following weights and data is based on the original data: \n")
for i in range(cols):
    print(Strings[i] + str(ogWeights[i]))
print("J-value: " + str(jValForOg) + "\n")

print("The following weights and data is based on the original data values squared")
for i in range(cols):
    print(Strings[i] + str(sqrdWeights[i]))
print("J-value: " + str(jValForSqrd) + "\n")

print("The following weights and data is based on the original and squared data :\n")
for i in range((cols * 2)):
    print(Strings12[i] + str(sqrdAndOgWeights[i]))
print("J-value: " + str(jValforOgAndSqrd) + "\n")

str1 = input("Please enter the filename of the validation file or test file: ")
inData = open(str1,"r")
str1 = inData.readline()
str1 = str1.split('\t')
rows2 = int(str1[0])
ogData2 = np.zeros([rows2, cols])
yValVector2 = np.zeros([rows2,1])
for i in range(rows2):
    str1 = inData.readline()
    line = str1.split("\t")
    for j in range(cols + 1):
        if(j == cols):
            yValVector2[i] = float(line[j])
        else:
            ogData2[i][j] = float(line[j])
inData.close()

sqrdData2 = np.zeros([rows2, cols])
for i in range(rows2):
    for j in range(cols):
        sqrdData2[i][j] = (ogData2[i][j]) * (ogData2[i][j])

ogAndsqrdData2 = np.zeros([rows2, (cols * 2)])
ogjCounter = 0
sqrdjCounter = 0
for i in range(rows2):
    ogjCounter = 0
    sqrdjCounter = 0
    for j in range((cols * 2)):
        if (j %  2 == 0):
            ogAndsqrdData2[i][j] = ogData2[i][ogjCounter]
            ogjCounter = ogjCounter + 1
        else:
            ogAndsqrdData2[i][j] = sqrdData2[i][sqrdjCounter]
            sqrdjCounter = sqrdjCounter + 1

jValForOgTest = jValCalc(ogData2,yValVector2,ogWeights,rows2)
jValForSqrdTest = jValCalc(sqrdData2,yValVector2,sqrdWeights,rows2)
jValforOgAndSqrdTest = jValCalc(ogAndsqrdData2,yValVector2,sqrdAndOgWeights,rows2)

ogRSquaredTest = calcRSquared(jValForOgTest,yValVector2,rows2)
sqrdRSquaredTest = calcRSquared(jValForSqrdTest,yValVector2,rows2)
ogAndSqrdRSquaredTest = calcRSquared(jValforOgAndSqrdTest,yValVector2,rows2)

AdOgRSquaredValTest = calcAdjustedRSquared(ogRSquaredTest, rows2, cols)
AdSqrdRSquaredValTest = calcAdjustedRSquared(sqrdRSquaredTest,rows2,cols)
AdOgAndSqrdRSquaredValTest = calcAdjustedRSquared(ogAndSqrdRSquaredTest,rows2,cols * 2)
    
print("\nThe J value when using the original equation with new data is: " + str(jValForOgTest))
print("The adjusted R squared value when using the original equation with new data is: " + str(AdOgRSquaredValTest))
print("\n")
print("The J value when using the squared equation with new data is: " + str(jValForSqrdTest))
print("The adjusted R squared value when using the original squared equation with new data is: " + str(AdSqrdRSquaredValTest))
print("\n")
print("The J value when using the original equation and the original squared equation with new data is: " + str(jValforOgAndSqrdTest))
print("The adjusted R squared value when using the original equation and the original squared equation with new data is: " + str(AdOgAndSqrdRSquaredValTest))
