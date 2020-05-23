from blackbox import BlackBox
import sys
import random
import time
start = time.time()
inputFile = sys.argv[1]
streamSize = int(sys.argv[2])
numOfAsks = int(sys.argv[3])
outputFile = sys.argv[4]

reservoirList = list()

def writeToFile(op):
    with open(outputFile, "w") as fw:
        fw.write(op)
        fw.close()

def reservoir(seqNum, data):
    global reservoirList
    count = seqNum
    if len(reservoirList) == 0:
        reservoirList = data
        count += len(data)
    else:
        for d in data:
            count += 1
            prob = random.randint(0, 100000) % count
            if (prob < streamSize):
                pos = random.randint(0, 100000) % streamSize
                reservoirList[pos] = d
    return str(count) + "," + str(reservoirList[0]) + "," + str(reservoirList[20]) + "," + str(reservoirList[40]) + "," + str(reservoirList[60]) + "," + str(reservoirList[80] + "\n")

if __name__ == "__main__":
    blackBox = BlackBox()
    random.seed(553)
    op = "seqnum,0_id,20_id,40_id,60_id,80_id\n"
    for i in range(numOfAsks):
        data = blackBox.ask(inputFile, streamSize)
        op += reservoir(i*streamSize, data)
    writeToFile(op)
    end = time.time()
    print("Duration:" + str(round(end - start,2)))
