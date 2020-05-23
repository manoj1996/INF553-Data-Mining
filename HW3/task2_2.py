import xgboost as xgb
import sys
from pyspark import SparkContext
import time
import  numpy as np
import json

start = time.time()
trainFolderPath, testFilePath, outputFilePath = sys.argv[1], sys.argv[2], sys.argv[3]
businesses_dict, businessId_dict = {}, {}
userId_dict, users_dict = {}, {}
sc = SparkContext(master="local[*]", appName="model-based-CF")
sc.setLogLevel("ERROR")
businessIdMap, userIdMap = {}, {}

def modelbasedCFRecommendation(trainRDD, testRDD):
	full_rdd = sc.union([trainRDD, testRDD])
	users_rdd, businesses_rdd  = full_rdd.map(lambda a:a[0]).distinct(), full_rdd.map(lambda a:a[1]).distinct()
	users, businesses = users_rdd.collect(), businesses_rdd.collect()
	nrows, ncols = len(users), len(businesses)
	for u in range(nrows):
		users_dict[users[u]] = u
		userId_dict[u] = users[u]
	for b in range(ncols):
		businesses_dict[businesses[b]] = b
		businessId_dict[b] = businesses[b]

	readUserJson(trainFolderPath+'/user.json')
	readBusinessJson(trainFolderPath+'/business.json')

	trainNorm = trainRDD.map(lambda x: (int(users_dict[x[0]]), int(businesses_dict[x[1]]), float(x[2])))
	testNorm = testRDD.map(lambda x: (int(users_dict[x[0]]), int(businesses_dict[x[1]])))
	testdata = np.array(testNorm.map(lambda x: [x[0], x[1]]).collect())
	trainY = np.array(trainNorm.map(lambda x: x[2]).collect())
	trainX = np.array(trainNorm.map(lambda x: [x[0], x[1]]).collect())
	tempX = list()
	for x in trainX:
		tempX.append([x[0], x[1], businessIdMap[x[1]][0], businessIdMap[x[1]][1], userIdMap[x[0]][0], userIdMap[x[0]][1]])
	trainX = np.array(tempX)
	testX = list()
	for test in testdata:
		testX.append([test[0], test[1], businessIdMap[test[1]][0], businessIdMap[test[1]][1], userIdMap[test[0]][0], userIdMap[test[0]][1]])

	testdata = np.array(testX)
	model = xgb.XGBRegressor(learning_rate = 0.1, n_estimators= 600, max_depth=5, min_child_weight=4, nthread=4)
	model.fit(trainX, trainY)
	output = model.predict(testdata)
	return testdata, output

def readBusinessJson(businessPath):
	businessDataset = sc.textFile(businessPath)
	businessRdd = businessDataset.map(json.loads).map(lambda x:(x['business_id'], x['stars'], x['review_count'])).collect()
	for x in businessRdd:
		if x[0] in businesses_dict:
			businessIdMap[businesses_dict[x[0]]] = [float(x[1]), int(x[2])]

def readUserJson(userPath):
	userDataset = sc.textFile(userPath)
	userRdd = userDataset.map(json.loads).map(lambda x:(x['user_id'], x['average_stars'], x['review_count'])).collect()
	for x in userRdd:
		if x[0] in users_dict:
			userIdMap[users_dict[x[0]]] = [float(x[1]), int(x[2])]

def writeToFile(fp, testData, predictedRatings):
	op = ""
	op += "user_id, business_id, prediction"
	for i in range(len(testData)):
		op += "\n"
		op += str(userId_dict[testData[i][0]]) + "," + str(businessId_dict[testData[i][1]]) + "," + str(predictedRatings[i])

	fp.write(op)
	fp.close()

trainRDD = sc.textFile(trainFolderPath+'/yelp_train.csv')\
    .map(lambda x : x.split(','))\
    .filter(lambda x: x[0]!= "user_id")\
    .persist()

testRDD = sc.textFile(testFilePath)\
    .map(lambda x : x.split(','))\
    .filter(lambda x: x[0]!= "user_id")\
    .persist()

testData, predictedRatings = modelbasedCFRecommendation(trainRDD, testRDD)
fp = open(outputFilePath, 'w')
writeToFile(fp, testData, predictedRatings)
end= time.time()
print("Duration: " + str(end-start))
