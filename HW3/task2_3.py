import xgboost as xgb
import sys

from keras.callbacks import EarlyStopping

from pyspark import SparkContext
import time
from operator import add
import  numpy as np
import json
from keras.models import Sequential
from keras.layers import Dense

start = time.time()
trainFolderPath, testFilePath, outputFilePath = sys.argv[1], sys.argv[2], sys.argv[3]
businesses_dict, businessId_dict = {}, {}
userId_dict, users_dict = {}, {}
sc = SparkContext(master="local[*]", appName="hybrid-based-CF-recommendation")
sc.setLogLevel("ERROR")
businessIdMap, userIdMap = {}, {}
userIdToBusinessIdToModel = {}
cosineSimilaritiesMatrix= {}
a = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
m = 0
numOfHashFunctions = len(a)
ind1 = -1
rowPerBand = 3
userIdTotalRating = {}
userIdTotalNum = {}
userIdAvgRating = {}
def getNormalizedRatings(x):
    user, ratings= x[0], x[1]
    sumRating = sum(list(map(lambda x: float(x), ratings)))
    avgRating= sumRating/len(ratings)
    newRatings = [float(i) for i in ratings]
    return (user,(newRatings,avgRating))

def createTuples(x):
    businessId, users, ratings, avgRating = x[0][0], x[0][1], x[1][0], x[1][1]
    rows = [((users[i], businessId), (ratings[i], avgRating)) for i in range(len(users))]
    return rows

def getCosineSimilarities(x, listBusinesseswiseUserRatings, keyedRatings):
    business1, business2 = x[0], x[1]
    if (business1 in listBusinesseswiseUserRatings and business2 in listBusinesseswiseUserRatings):
        userlist1, userlist2 = listBusinesseswiseUserRatings[business1][0], listBusinesseswiseUserRatings[business2][0]
        coRatedUsers = list(set(userlist1).intersection(set(userlist2)))
        global N2
        global cosineSimilaritiesMatrix
        count = 0
        numerator, denominator1, denominator2, cosineSimilarity = 0, 0, 0, 0
        j = 0
        while ((j < len(coRatedUsers)) & (count < N2)):
            rating1, rating2 = float(keyedRatings[(coRatedUsers[j], business1)][0]), float(keyedRatings[(coRatedUsers[j], business2)][0])
            numerator = numerator + rating1 * rating2
            denominator1 += rating1**2
            denominator2 += rating2**2
            count += 1
            j += 1

        if (numerator != 0):
            cosineSimilarity = float(numerator / (denominator1 * denominator2)**0.5)
        cosineSimilaritiesMatrix[x] = cosineSimilarity
        return cosineSimilarity
    return 0

def minhashing(x):
    global a, m
    minNum = list()
    for y in a:
        minNumSub = list()
        for k in x[1]:
            minNumSub.append(((y * k) + 1) % m)
        minNum.append(min(minNumSub))
    return (x[0], minNum)

def intermediate_step1(x):
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

def get_candidates(x):
    businesses = sorted(x[1])
    candidates = list()
    businessLength = len(businesses)
    m = 0
    while (m < businessLength):
        n = m + 1
        while (n < businessLength):
            candidates.append(((businesses[m], businesses[n]), 1))
            n += 1
        m += 1
    return candidates

def find_jaccard_similarity(x, businessWiseUsers):
    business1, business2 = x[0][0], x[0][1]
    users1, users2 = set(businessWiseUsers[business1]), set(businessWiseUsers[business2])
    jaccardSimilarity = float(len(users1 & users2) / len(users1 | users2))
    return (((business1, business2), jaccardSimilarity))

def get_similar_businesses_lsh(train_rdd):
    users = train_rdd.map(lambda a: a[0])\
        .distinct()\
        .collect()
    businesses = train_rdd.map(lambda a: a[1])\
        .distinct()\
        .collect()
    nrows, ncols = len(users), len(businesses)
    businesswise_users = train_rdd.map(lambda x: (x[1], [x[0]]))\
        .reduceByKey(add)\
        .collectAsMap()

    userDict = {}
    u, b = 0, 0
    while (u < nrows):
        userDict[users[u]] = u
        u += 1
    businesses_dict = {}
    while (b < ncols):
        businesses_dict[businesses[b]] = b
        b += 1
    global m
    m = nrows
    characteristic_matrix = train_rdd.map(lambda r: (r[1], [userDict[r[0]]])) \
        .reduceByKey(add)
    sig = characteristic_matrix.map(lambda x: minhashing(x))\
        .flatMap(lambda x: intermediate_step1(x))
    candidate_gen = sig.reduceByKey(add)\
        .filter(lambda x: len(x[1]) > 1)
    candidates = candidate_gen.flatMap(lambda x: get_candidates(x)) \
        .distinct()
    jaccard_similarity_rdd = candidates.map(lambda x: find_jaccard_similarity(x, businesswise_users))\
        .filter(lambda x: x[1] >= 0.5)
    return jaccard_similarity_rdd

def getImprovedItemRating(x, cosines_matrix, userwise_businesses, list_businesseswise_user_ratings, keyed_ratings):
    user_id, business_id, businesses = x[0], x[1], userwise_businesses[x[0]]
    global N3
    if (business_id in cosines_matrix):
        businesses = list(set(businesses).intersection(set(cosines_matrix[business_id])))
    count = 0
    cosineSimilarities = []
    backup = list()
    b = 0
    while(b < len(businesses)):
        if (businesses[b] != business_id):
            if ((business_id, businesses[b]) in cosineSimilaritiesMatrix):
                cosine_similarity = cosineSimilaritiesMatrix[(business_id, businesses[b])]
            elif ((businesses[b], business_id) in cosineSimilaritiesMatrix):
                cosine_similarity = cosineSimilaritiesMatrix[(businesses[b], business_id)]
            elif (business_id in list_businesseswise_user_ratings and businesses[b] in list_businesseswise_user_ratings):
                cosine_similarity = getCosineSimilarities((business_id, businesses[b]), list_businesseswise_user_ratings,
                                                            keyed_ratings)
            else:
                cosine_similarity = 0

            backup.append((businesses[b], cosine_similarity))
            if (cosine_similarity >= 1):
                cosineSimilarities.append((businesses[b], cosine_similarity))
                count = count + 1
            if (count >= N3):
                break
        b += 1

    if (len(backup) < N3):
        cosineSimilarities = backup

    num, den = 0, 0
    i = 0
    while (i < len(cosineSimilarities)):
        business_id2 = cosineSimilarities[i][0]
        cosine_similarity = cosineSimilarities[i][1]
        if ((user_id, business_id2) in keyed_ratings):

            rating = float(keyed_ratings[(user_id, business_id2)][0])
            num += float(rating * cosine_similarity)
            den += float(abs(cosine_similarity))
        i += 1
    if (den != 0 and (num > 400 and den > 200)):
        return ((x[0], x[1]), num / den)
    return ((x[0], x[1]), float(userIdToBusinessIdToModel[users_dict[x[0]]][businesses_dict[x[1]]][0]))

def itemBasedCFRecommendation(train_rdd, test_rdd):
    test_pairs = test_rdd.map(lambda x: (x[0], x[1]))
    userwise_businesses = train_rdd.map(lambda x: (x[0], [x[1]]))\
        .reduceByKey(add)\
        .collectAsMap()
    normalized = train_rdd.map(lambda x: (x[1], [x[2]]))\
        .reduceByKey(add)\
        .map(lambda x: getNormalizedRatings(x))
    businesswise_user_ratings = train_rdd.map(lambda x: (x[1], [x[0]]))\
        .reduceByKey(add)\
        .join(normalized) \
        .map(lambda x: ((x[0], x[1][0]), (x[1][1][0], x[1][1][1])))

    keyed_ratings = businesswise_user_ratings.flatMap(lambda x: createTuples(x))\
        .collectAsMap()
    list_businesseswise_user_ratings = businesswise_user_ratings.map(lambda x: (x[0][0], (x[0][1], x[1][0], x[1][1])))\
        .collectAsMap()

    similar_businesses = get_similar_businesses_lsh(train_rdd).distinct()
    cosines_matrix = similar_businesses.flatMap(lambda x: (((x[0][0], x[0][1]), x[1]), ((x[0][1], x[0][0]), x[1])))\
        .map(lambda x: (x[0], [x[1]]))\
        .reduceByKey(add)\
        .collectAsMap()
    predictedRatings = test_pairs.map(lambda x: getImprovedItemRating(x, cosines_matrix, userwise_businesses, list_businesseswise_user_ratings,
                                           keyed_ratings))
    return predictedRatings

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
    trainY = np.array(trainNorm.map(lambda x: x[2]).collect())
    trainX = np.array(trainNorm.map(lambda x: [x[0], x[1]]).collect())
    testNorm = testRDD.map(lambda x: (int(users_dict[x[0]]), int(businesses_dict[x[1]])))
    testdata = np.array(testNorm.map(lambda x: [x[0], x[1]]).collect())
    tempX = list()
    for x in trainX:
        tempX.append([x[0], x[1], businessIdMap[x[1]][0], businessIdMap[x[1]][1], userIdMap[x[0]][0], userIdMap[x[0]][1]])
    trainX = np.array(tempX)
    testX = list()
    for test in testdata:
        testX.append([test[0], test[1], businessIdMap[test[1]][0], businessIdMap[test[1]][1], userIdMap[test[0]][0],
                      userIdMap[test[0]][1]])

    testdata = np.array(testX)
    model = xgb.XGBRegressor(learning_rate = 0.1, n_estimators= 600, max_depth=5, min_child_weight=4, nthread=4)
    model.fit(trainX, trainY)
    output = model.predict(testdata)
    i = 0
    for m in testdata:
        if m[0] not in userIdToBusinessIdToModel:
            userIdToBusinessIdToModel[m[0]] = {}
        if m[1] not in userIdToBusinessIdToModel[m[0]]:
            userIdToBusinessIdToModel[m[0]][m[1]] = []
        userIdToBusinessIdToModel[m[0]][m[1]].append(output[i])
        i += 1

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

def deepLearningRecommender(trainRDD, testRDD):

    full_rdd = sc.union([trainRDD, testRDD])
    users_rdd, businesses_rdd = full_rdd.map(lambda a: a[0]).distinct(), full_rdd.map(lambda a: a[1]).distinct()
    users, businesses = users_rdd.collect(), businesses_rdd.collect()
    nrows, ncols = len(users), len(businesses)

    for u in range(nrows):
        users_dict[users[u]] = u
        userId_dict[u] = users[u]

    for b in range(ncols):
        businesses_dict[businesses[b]] = b
        businessId_dict[b] = businesses[b]

    readUserJson(trainFolderPath + '/user.json')
    readBusinessJson(trainFolderPath + '/business.json')
    trainNorm = trainRDD.map(lambda x: (int(users_dict[x[0]]), int(businesses_dict[x[1]]), float(x[2])))
    trainY = np.array(trainNorm.map(lambda x: x[2]).collect())
    trainX = np.array(trainNorm.map(lambda x: [x[0], x[1]]).collect())
    testNorm = testRDD.map(lambda x: (int(users_dict[x[0]]), int(businesses_dict[x[1]])))
    testdata = np.array(testNorm.map(lambda x: [x[0], x[1]]).collect())
    tempX = list()
    for x in trainX:
        tempX.append(
            [x[0], x[1], businessIdMap[x[1]][0], businessIdMap[x[1]][1], userIdMap[x[0]][0], userIdMap[x[0]][1]])
    trainX = np.array(tempX)
    testX = list()
    for test in testdata:
        testX.append([test[0], test[1], businessIdMap[test[1]][0], businessIdMap[test[1]][1], userIdMap[test[0]][0],
                      userIdMap[test[0]][1]])

    testdata = np.array(testX)
    n_cols = len(testdata[0])
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stopping_monitor = EarlyStopping(patience=3)
    model.fit(trainX, trainY, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
    test_y_predictions = model.predict(testdata)

    print(test_y_predictions)


def writeToFile(fp, predictedRatings):
    op = ""
    op += "user_id, business_id, prediction"
    predictedRatingsAction = predictedRatings.collect()
    for i in predictedRatingsAction:
        op += "\n"
        op += str(i[0][0]) + "," + str(i[0][1]) + "," + str(i[1])
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

N2=4000
N3=4000

# modelbasedCFRecommendation(trainRDD, testRDD)
# predictedRatingsItem = itemBasedCFRecommendation(trainRDD, testRDD)
output = deepLearningRecommender(trainRDD, testRDD)
# fp = open(outputFilePath, 'w')
writeToFile(fp, predictedRatingsItem)
end= time.time()
print("Duration: " + str(end-start))
