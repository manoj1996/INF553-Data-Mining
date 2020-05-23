import sys
from pyspark import SparkContext

inputFilePath1 = sys.argv[1]
inputFilePath2 = sys.argv[2]
sc = SparkContext("local[*]", appName="ComputeRMSE")
rmseKeys = {}
def computeBuckets(rdd):
    rdd2 = rdd.map(lambda x: (str(x[0])+str(x[1]), x[2])).collect()
    for x,y in rdd2:
        if x not in rmseKeys:
            rmseKeys[x] = 0
        rmseKeys[x] = y

rdd1 = sc.textFile(inputFilePath1)\
    .map(lambda x: x.split(','))\
    .filter(lambda x: x[0] != "user_id")\
    .persist()

rdd2 = sc.textFile(inputFilePath2)\
    .map(lambda x: x.split(','))\
    .filter(lambda x: x[0] != "user_id")\
    .map(lambda x: (str(x[0])+str(x[1]), x[2])).collect()

computeBuckets(rdd1)

total = 0
for i,j in rdd2:
    total += (float(j)-float(rmseKeys[i]))**2
print(total)
print(len(rmseKeys))
rmse = (float(total/len(rmseKeys)))**0.5

print("rmse:", rmse)
