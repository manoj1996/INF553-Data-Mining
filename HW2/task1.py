from pyspark import SparkContext
from operator import add
import sys
import time
import itertools

startTime = time.time()
sc = SparkContext(master="local[*]", appName="user-business-frequent-itemsets")
sc.setLogLevel("ERROR")
inputFilePath = sys.argv[3]
outputFilePath = sys.argv[4]
caseNo = int(sys.argv[1])
support = int(sys.argv[2])

def makeBasket(rdd) :
    if caseNo==1:
        basket = rdd.map(lambda a: (a[0], [a[1]]))\
            .reduceByKey(add)\
            .persist()\
            .filter(lambda x: x[0]!= "user_id")\
            .map(lambda x: x[1])
    else :
        basket = rdd.map(lambda a:(a[1], a[0]))\
            .map(lambda a: (a[0], [a[1]]))\
            .reduceByKey(add)\
            .persist()\
            .filter(lambda x: x[0]!= "business_id")\
            .map(lambda x: x[1])
    return basket
# def customPartition(r):
#     return hash(str(r[0]))
dataRdd = sc.textFile(inputFilePath)\
    .map(lambda row: row.split(","))\
    .map(lambda r: (r[0], r[1]))\
    .distinct()\
    .persist()

# dataRdd = dataRdd.partitionBy(20, customPartition)
basket = makeBasket(dataRdd)
# print(basket.take(10))
numberOfPartitions = dataRdd.getNumPartitions()
scaledSupport = support/numberOfPartitions

itemList = list(set().union(*basket.collect()))
# print(itemList)

def getCandidateItemsets(itemset, k):
    cand = []
    m = 0
    while m < len(itemset)-1:
        n = m + 1
        while n < len(itemset):
            if itemset[m][:k-2] == itemset[n][:k-2]:
                c = list(set(itemset[m]).union(set(itemset[n])))
                c = sorted(c)
                if c not in cand:
                    cand.append(c)
            else:
                break
            n += 1
        m += 1
    return cand

def countOccurrence(setA, setB):
    setA = set(setA)
    return len(list(filter(lambda x: setA.issubset(x), setB)))

def countOccurrencePhase2(listOfItemsInBasket, itemSets):
    frequencyList = []
    buckets = list(listOfItemsInBasket)

    for i in itemSets:
        count = 0
        for b in buckets:
            if set(i).issubset(b):
                count += 1
        frequencyList.append([i, count])
    return frequencyList


def apriori(listOfItemsInBasket, itemSet, scaledSupport):
    frequentItemSet = []
    buckets = list(listOfItemsInBasket)
    Construct = []
    Filter = []
    for item in itemSet:
        count = 0
        for b in buckets:
            if item in set(b):
                count += 1
        Construct.append(item)
        if count >= scaledSupport:
            Filter.append(item)
    Filter = sorted(Filter)
    itemLength = len(Filter)
    Filter1 = [(x,) for x in Filter]
    frequentItemSet.extend(Filter1)

    Construct = list()
    for x in itertools.combinations(Filter, 2):
        Construct.append(sorted(list(x)))

    Construct = sorted(Construct)
    Filter[:] = []
    for i in Construct:
        count = countOccurrence(i, buckets)
        if count >= scaledSupport:
            Filter.append(i)
    Filter = sorted(Filter)

    frequentItemSet.extend(Filter)

    itemSetSize = 3
    for k in range(itemSetSize, itemLength):
        Construct[:] = []
        Construct = getCandidateItemsets(Filter, itemSetSize)
        if len(Construct) == 0:
            break
        Construct = sorted(Construct)
        Filter[:] = []

        for c in Construct:
            count = countOccurrence(c, buckets)
            if count >= scaledSupport:
                Filter.append(c)
        Filter = sorted(Filter)
        frequentItemSet.extend(Filter)
        itemSetSize += 1

    return frequentItemSet

def writeCandidate(fp, SONReduce1):
    fp.write("Candidates:\n")
    for l in range(1, len(itemList)):
        op = ""
        index = 0
        while index < len(SONReduce1):
            if len(SONReduce1[index]) == l:
                op += str(SONReduce1[index])
            index += 1

        op = op.replace(")(", "),(").replace(",)", ")")
        if op != "":
            if l != 1:
                fp.write("\n\n")
            fp.write(op)
    fp.write("\n\n")

def writeFrequent(fp, SONReduce2):
    fp.write("Frequent Itemsets:\n")
    for l in range(1, len(itemList)):
        op = ""
        index = 0
        while index < len(SONReduce2):
            if len(SONReduce2[index][0]) == l:
                op += str(SONReduce2[index][0])
            index += 1
        if op == "":
            break
        else:
            if l != 1:
                fp.write("\n\n")
            op = op.replace(")(", "),(").replace(",)", ")")
            fp.write(op)

#SON Algorithm Phase 1
SONMap1 = basket.mapPartitions(lambda y: apriori(y, itemList, scaledSupport))\
    .map(lambda x: (tuple(x),1))
SONReduce1 = SONMap1.distinct()\
    .sortByKey()\
    .map(lambda x: x[0])\
    .collect()


SONMap2 = basket.mapPartitions(lambda b: countOccurrencePhase2(b, SONReduce1))
SONReduce2 = SONMap2.reduceByKey(add)\
    .filter(lambda a: a[1]>=support)\
    .sortByKey()\
    .collect()

fw = open(outputFilePath, "w")
writeCandidate(fw, SONReduce1)
writeFrequent(fw, SONReduce2)
fw.close()
endTime = time.time()
print("Duration: ", str(round(endTime-startTime, 4)))