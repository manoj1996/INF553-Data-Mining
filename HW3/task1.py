import sys
from pyspark import SparkContext
from operator import add
import time
import random

start = time.time()
inputFilePath = sys.argv[1]
outputFilePath = sys.argv[2]
sc = SparkContext(master="local[*]", appName="jaccard_based_LSH")
sc.setLogLevel("ERROR")
ind = -1
ind1 = -1
ind2 = -1
rowPerBand = 3

def generateRandomCoeff(k, nrows):
    coeffsList = set()
    for i in range(k):
        randRow = random.randint(0, nrows)
        while randRow in coeffsList:
            randRow = random.randint(0, nrows)
        coeffsList.add(randRow)
    return coeffsList

def minhashing(x):
    global a, m
    minNum = list()
    for y in a:
        minNumSub = list()
        for k in x[1]:
            minNumSub.append(((y * k) + 1) % m)
        minNum.append(min(minNumSub))
    return (x[0], minNum)

def getBusinessSignature(x):
    global ind
    ind += 1
    return tuple((businesses[ind], x))

def intermediateStep1(x):
    global ind1
    global rowPerBand
    bands = int(len(x[1]) / rowPerBand)

    ind1 += 1
    businessId = x[0]
    signaturesList = x[1]

    bandsList = list()
    rowindex = 0
    for band in range(bands):
        row = list()
        for rpb in range(rowPerBand):
            row.append(signaturesList[rowindex])
            rowindex = rowindex + 1
        bandsList.append(((band, tuple(row)), [businessId]))
        row[:] = []
    return bandsList

def getCandidates(x):
    businesses = sorted(x[1])
    candidates = list()
    businessLength = len(businesses)
    m = 0
    while (m < businessLength):
        n = m+1
        while(n < businessLength):
            candidates.append(((businesses[m], businesses[n]), 1))
            n += 1
        m += 1
    return candidates

def findJaccardSimilarity(x):
    business1, business2 = x[0][0], x[0][1]
    users1, users2 = set(businessWiseUsers[business1]), set(businessWiseUsers[business2])
    jaccardSimilarity = float(len(users1 & users2) / len(users1 | users2))
    return (((business1, business2), jaccardSimilarity))

def indexBusiness(x):
    global ind2
    ind2 = ind2 + 1
    return ((x[0], ind2))

def writeToFile(fp, sorted_js_rdd):
    op = ""
    op += "business_id_1, business_id_2, similarity"
    js_rdd_action = sorted_js_rdd.collect()
    for i in js_rdd_action:
        op += "\n"
        op += str(i[0]) + "," + str(i[1][0]) + "," + str(i[1][1])
    fp.write(op)
    fp.close()

rdd = sc.textFile(inputFilePath)\
    .map(lambda x: x.split(','))\
    .filter(lambda x: x[0] != "user_id").persist()

businessWiseUsers = rdd.map(lambda x: (x[1], [x[0]])).reduceByKey(add).collectAsMap()

users = rdd.map(lambda a: a[0])\
    .distinct()\
    .collect()
businesses = rdd.map(lambda a: a[1])\
    .distinct()\
    .collect()

nrows, ncols = len(users), len(businesses)
userDict = {}
u,b = 0, 0
while(u < nrows):
    userDict[users[u]] = u
    u += 1

businesses_dict = {}
while(b < ncols):
    businesses_dict[businesses[b]] = b
    b += 1

characteristic_matrix = rdd.map(lambda r: (r[1], [userDict[r[0]]]))\
    .reduceByKey(add)

a = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
m = nrows
signature_matrix = characteristic_matrix.map(lambda x: minhashing(x))
sig = signature_matrix.flatMap(lambda x: intermediateStep1(x))
candidate_gen = sig.reduceByKey(add)\
    .filter(lambda x: len(x[1]) > 1)
candidates = candidate_gen.flatMap(lambda x: getCandidates(x))\
    .distinct()
jaccard_similarity_rdd = candidates.map(lambda x: findJaccardSimilarity(x))\
    .filter(lambda x: x[1] >= 0.5)
sorted_js_rdd = jaccard_similarity_rdd.map(lambda x: (x[0][1], (x[0][0], x[1])))\
    .sortByKey()\
    .map(lambda x: (x[1][0], (x[0], x[1][1])))\
    .sortByKey()

fp = open(outputFilePath, 'w')
writeToFile(fp, sorted_js_rdd)
end = time.time()
print("Duration: " + str(end - start))