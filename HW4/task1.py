from pyspark import SQLContext, SparkContext
from pyspark.sql.functions import *
import time
import sys
import os
from pyspark.sql.types import *
from graphframes import *
from pyspark.sql import Row

start = time.time()
os.environ["PYSPARK_SUBMIT_ARGS"] = ( "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")
sc = SparkContext(master="local[*]", appName="graphFrames-community-detection")
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")

inputFile = sys.argv[1]
outputFile = sys.argv[2]

inpData = sc.textFile(inputFile)
rddData = inpData.map(lambda x : x.split(' '))
rdd = rddData.union(inpData.map(lambda x : x.split(' ')[::-1]))\
	.persist()

edgeSchema = StructType([StructField('src', StringType()), StructField('dst',StringType())])
vertexSchema = StructType([StructField('id', StringType())])
graphVertices= rdd.flatMap(lambda v: [Row(v[0]),Row(v[1])])\
	.distinct()
graphEdges = rdd.map(lambda x: (x[0], x[1]))\
	.map(lambda x: Row(src=x[0], dst=x[1]))
verticesDataframe= sqlContext.createDataFrame(graphVertices, vertexSchema)
edgesDataframe = sqlContext.createDataFrame(graphEdges, edgeSchema)

graph = GraphFrame(verticesDataframe, edgesDataframe)
graphList = graph.labelPropagation(maxIter=5)\
	.rdd\
	.map(lambda x: (x[1], x[0]))\
	.groupByKey()\
	.map(lambda x: x[1])\
	.collect()

sortedGraph = list()
for i in range(len(graphList)):
	j = sorted(graphList[i])
	k = len(j)
	sortedGraph.append((j, k))
sortedGraph.sort(key=lambda x: x[0])
sortedGraph.sort(key=lambda x: x[1])
with open(outputFile, 'w') as fw:
	op = ""
	for i in sortedGraph:
		op += str(i[0]).replace("]","").replace("[","") + "\n"
	fw.write(str(op))
	fw.close()

end = time.time()
print("Duration: "+str(end-start))