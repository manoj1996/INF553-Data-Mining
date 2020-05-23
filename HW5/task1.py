from blackbox import BlackBox
import sys
import binascii
import random
import time
start = time.time()
inputFile, streamSize = sys.argv[1], int(sys.argv[2])
numOfAsks, outputFile = int(sys.argv[3]), sys.argv[4]
numOfHashFunctions = 80
filterBitArray = [0]*69997
m = len(filterBitArray)
actualUsers, predictedUsers = set(), set()

FP, TN = 0, 0

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

def bloom_filtering(time, data):
    global filterBitArray
    global hashFunctions
    global actualUsers
    global predictedUsers
    global m
    global FP
    global TN
    for i in data:
        hashvals = myhashs(i)
        ctr = sum([1 if filterBitArray[j] == 1 else 0 for j in hashvals])
        if(i not in actualUsers):
            if(ctr == len(hashvals)):
                predictedUsers.add(i)
                FP += 1
            else:
                TN += 1

        for i in hashvals:
            if(filterBitArray[i] != 1):
                filterBitArray[i] = 1
        actualUsers.add(i)

    FPR = float(FP)/(FP+TN)
    return "\n"+str(time)+","+str(FPR)

def writeToFile(op):
    with open(outputFile, "w") as fw:
        fw.write(op)
        fw.close()
def main():
    blackBox = BlackBox()
    op = "Time,FPR"
    for i in range(numOfAsks):
        stream_users = blackBox.ask(inputFile, streamSize)
        op += bloom_filtering(i, stream_users)
    writeToFile(op)

if __name__ == "__main__":
    main()
    end = time.time()
    print("Duration:" + str(round(end - start,2)))