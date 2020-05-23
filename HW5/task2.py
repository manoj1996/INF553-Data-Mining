from blackbox import BlackBox
import sys
import binascii
import random
import math
import time
start = time.time()
groundTruths = 0
estimator = 0
numOfHashFunctions = 200
num_groups = 25
m= 5000
inputFile, streamSize = sys.argv[1], int(sys.argv[2])
numOfAsks, outputFile = int(sys.argv[3]), sys.argv[4]
rows_per_group= int(numOfHashFunctions/num_groups)

hashFunctions = list()
for i in range(numOfHashFunctions):
    hashFunctions.append([random.randint(1,100), random.randint(1,100)])

def myhashs(s):
    global hashFunctions
    global m
    result = list()
    intUser = int(binascii.hexlify(s.encode('utf8')), 16)
    for f in hashFunctions:
        result.append((f[0] * intUser + f[1]) % m)
    return result

def flajolet_martin(time, data):
    global numOfHashFunctions
    global num_groups
    global rows_per_group
    global hash_functions
    global m
    global estimator
    global groundTruths

    def trailing_zeroes(mystr):
        return len(mystr) - len(mystr.rstrip('0'))

    actualUsers, hashvals, hashvals_bin, estimates, groupwise_averages = set(), list(), list(), list(), list()

    for i in data:
        actualUsers.add(i)
        h1 = myhashs(i)
        h2= list()
        for i in h1:
            h2.append(bin(i)[2:])

        hashvals_bin.append(h2)
        hashvals.append(h1)

    for i in range(numOfHashFunctions):
        maxval = -1
        j = 0
        while(j < len(hashvals_bin)):
            z = trailing_zeroes(hashvals_bin[j][i])
            if(z > maxval):
                maxval= z
            j += 1
        powR = math.pow(2,maxval)
        estimates.append(powR)

    for i in range(num_groups):
        avg = 0
        j = 0
        while(j < rows_per_group):
            avg += estimates[i*rows_per_group + j]
            j += 1

        avg= round(avg/rows_per_group)
        groupwise_averages.append(avg)

    groupwise_averages.sort()
    predicted_num_distinct= groupwise_averages[int(num_groups/2)]
    groundTruths += len(actualUsers)
    estimator += predicted_num_distinct

    return "\n"+str(time)+","+str(int(len(actualUsers)))+","+str(int(predicted_num_distinct))

def writeToFile(op):
    with open(outputFile, "w") as fw:
        fw.write(op)
        fw.close()

def main():
    blackBox = BlackBox()
    op = "Time,Ground Truth,Estimation"
    for i in range(numOfAsks):
        stream_users = blackBox.ask(inputFile, streamSize)
        op += flajolet_martin(i, stream_users)
    writeToFile(op)
if __name__ == "__main__":
    main()
    print("Ratio(sum of estimates/sum of ground truths):"+str(round(float(estimator/groundTruths), 2)))
    end = time.time()
    print("Duration:" + str(round(end - start,2)))