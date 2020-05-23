import sys
import time
from pyspark import SparkContext
from operator import add

start = time.time()
trainFilePath = sys.argv[1]
testFilePath = sys.argv[2]
outputFilePath = sys.argv[3]
cosineSimilaritiesMatrix= {}
a = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
m = 0
numOfHashFunctions = len(a)
ind1 = -1
rowPerBand = 3
userIdTotalRating = {}
userIdTotalNum = {}
userIdAvgRating = {}
sc = SparkContext(master="local[*]", appName="item-based-CF")
sc.setLogLevel("ERROR")

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
        if cosineSimilarity == 0:
            cosineSimilarity = 0.1
        cosineSimilaritiesMatrix[x] = cosineSimilarity
        return cosineSimilarity
    return 0.1

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
                cosine_similarity = list_businesseswise_user_ratings[businesses[b]][2]

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

    if (den != 0):
        return ((x[0], x[1]), num / den)
    if user_id in userIdAvgRating:
        return ((x[0], x[1]), userIdAvgRating[user_id])
    return ((x[0], x[1]), 0)

def computeUserAvg(rdd):
    users = rdd.collect()
    for u, uListIterator in users:
        for uList in uListIterator:
            if uList[0] not in userIdTotalRating:
                userIdTotalRating[uList[0]] = 0
                userIdTotalNum[uList[0]] = 0
            userIdTotalRating[uList[0]] += float(uList[1])
            userIdTotalNum[uList[0]] += 1
    for u in userIdTotalRating:
        userIdAvgRating[u] = float(userIdTotalRating[u] / userIdTotalNum[u])

def itemBasedCFRecommendation(train_rdd, test_rdd):
    test_pairs = test_rdd.map(lambda x: (x[0], x[1]))
    userwise_businesses = train_rdd.map(lambda x: (x[0], [x[1]]))\
        .reduceByKey(add)\
        .collectAsMap()
    normalized = train_rdd.map(lambda x: (x[1], [x[2]]))\
        .reduceByKey(add)\
        .map(lambda x: getNormalizedRatings(x))
    avgUserrdd = train_rdd.map(lambda x: (x[0], x[2])).groupBy(lambda x: x[0])
    computeUserAvg(avgUserrdd)
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

def writeToFile(fp, predictedRatings):
    op = ""
    op += "user_id, business_id, prediction"
    predictedRatingsAction = predictedRatings.collect()
    for i in predictedRatingsAction:
        op += "\n"
        op += str(i[0][0]) + "," + str(i[0][1]) + "," + str(i[1])
    fp.write(op)
    fp.close()

trainRDD = sc.textFile(trainFilePath)\
    .map(lambda x : x.split(','))\
    .filter(lambda x: x[0]!= "user_id")\
    .persist()

testRDD = sc.textFile(testFilePath)\
    .map(lambda x : x.split(','))\
    .filter(lambda x: x[0]!= "user_id")\
    .persist()

N2=4000
N3=4000
predictedRatings = itemBasedCFRecommendation(trainRDD, testRDD)
fp = open(outputFilePath, 'w')
writeToFile(fp, predictedRatings)
end= time.time()
print("Duration: " + str(end-start))