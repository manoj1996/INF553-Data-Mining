from __future__ import print_function
import sys
from pyspark import SparkConf, SparkContext
import time
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
import math
import itertools

start = time.time()
cosine_similarities_matrix = {}
a = [1, 3, 9, 11, 13, 17, 19, 27, 29, 31, 33, 37, 39, 41, 43, 47, 51, 53, 57, 59]
m = 0

num_of_hash_functions = 20
index1 = -1
row_per_band = 2


# ================================================ USER BASED COLLABORATIVE FILTERING ================================================

def get_normalised_ratings(x):
    user = x[0]
    ratings = x[1]

    avg_rating = 0

    for i in ratings:
        avg_rating = avg_rating + float(i)
    avg_rating = avg_rating / len(ratings)

    new_ratings = []
    for i in ratings:
        new_ratings.append(float(i))
    # new_ratings.append(float(i)-avg_rating)
    return (user, (new_ratings, avg_rating))

# ================================================ ITEM BASED COLLABORATIVE FILTERING ================================================
def convert2(x):
    business_id = x[0][0]
    users = x[0][1]
    ratings = x[1][0]
    avg_rating = x[1][1]

    rows = []
    for i in range(0, len(users)):
        row = ((users[i], business_id), (ratings[i], avg_rating))
        rows.append(row)
    return rows


def get_cosine_similarities(x, list_businesseswise_user_ratings, keyed_ratings):
    business1 = x[0]
    business2 = x[1]
    # print(business1)
    # print(business2)

    if (business1 in list_businesseswise_user_ratings and business2 in list_businesseswise_user_ratings):
        userlist1 = list_businesseswise_user_ratings[business1][0]
        userlist2 = list_businesseswise_user_ratings[business2][0]

        ratinglist1 = list_businesseswise_user_ratings[business1][1]
        ratinglist2 = list_businesseswise_user_ratings[business2][1]
        # print(str(userlist1))
        # print(str(userlist2))

        avg_rating1 = list_businesseswise_user_ratings[business1][2]
        avg_rating2 = list_businesseswise_user_ratings[business2][2]

        co_rated_users = set(userlist1) & set(userlist2)

        global N2
        count = 0
        numerator = 0
        denominator1 = 0
        denominator2 = 0
        cosine_similarity = 0
        for j in co_rated_users:
            if (count < N2):
                rating1 = float(keyed_ratings[(j, business1)][0])
                # -avg_rating1
                rating2 = float(keyed_ratings[(j, business2)][0])
                # -avg_rating2
                # print("rating1: "+str(rating1))
                # print("rating2: "+str(rating2))

                numerator = numerator + rating1 * rating2
                denominator1 = denominator1 + rating1 * rating1
                denominator2 = denominator2 + rating2 * rating2
                count = count + 1
            else:
                break

        if (numerator != 0):
            # print("Numerator: "+str(numerator))
            # print("denominator1: "+str(denominator1))
            # print("denominator2: "+str(denominator2))
            cosine_similarity = float(numerator / math.sqrt(denominator1 * denominator2))
        # print("cosine_similarity: "+str(cosine_similarity))
        # print("\n")

        global cosine_similarities_matrix
        cosine_similarities_matrix[x] = cosine_similarity

        return cosine_similarity
    else:
        return 0


def get_item_rating(x, userwise_businesses, list_businesseswise_user_ratings, keyed_ratings):
    user_id = x[0]
    business_id = x[1]

    # -------------------------------- Calculating Weighted Average --------------------------------
    weighted_average = 0
    num = 0
    den = 0

    businesses = userwise_businesses[user_id]
    # if(business_id in list_businesseswise_user_ratings):
    # 	users_rated= list_businesseswise_user_ratings[business_id][0]
    # 	for u in users_rated:
    # 		businesses.extend(userwise_businesses[u])
    # 	businesses= list(set(businesses))
    count = 0
    cosine_similarities = []
    backup = []
    for b in businesses:
        cosine_similarity = 0
        if (b != business_id):
            if ((business_id, b) in cosine_similarities_matrix):
                cosine_similarity = cosine_similarities_matrix[(business_id, b)]
            # print("A")
            elif ((b, business_id) in cosine_similarities_matrix):
                cosine_similarity = cosine_similarities_matrix[(b, business_id)]
            # print("B")
            elif (business_id in list_businesseswise_user_ratings and b in list_businesseswise_user_ratings):
                cosine_similarity = get_cosine_similarities((business_id, b), list_businesseswise_user_ratings,
                                                            keyed_ratings)
            # print("C")
            else:
                cosine_similarity = 0

            backup.append((b, cosine_similarity))
            if (cosine_similarity >= 1):
                cosine_similarities.append((b, cosine_similarity))
                count = count + 1

            if (count >= N2):
                break

    if (len(backup) < N2):
        cosine_similarities = backup

    # cosine_similarities.sort(key=lambda x: x[1], reverse=True)

    # global N2
    # cosine_similarities= cosine_similarities[:N2]

    num = 0
    den = 0
    for i in cosine_similarities:
        business_id2 = i[0]
        cosine_similarity = i[1]

        if ((user_id, business_id2) in keyed_ratings):
            avg_rating = float(keyed_ratings[(user_id, business_id2)][1])
            # print(business_id2+" : "+str(cosine_similarity)+" : "+str(avg_rating))

            rating = float(keyed_ratings[(user_id, business_id2)][0])
            # -avg_rating

            # print(str(rating)+", "+str(cosine_similarity))
            num = num + float(rating * cosine_similarity)
            den = den + float(abs(cosine_similarity))

    if (den == 0):
        return ((x[0], x[1]), 0)
    else:

        # print("RATING: "+str(num)+"/"+str(den))
        rt = num / den
        # print("Predicted score :"+str(rt))

        score = ((x[0], x[1]), rt)
        return score


def itembased_cf_recommendation(train_rdd, test_rdd):
    businesses = train_rdd.map(lambda x: x[1]).distinct().collect()

    # list_businesseswise_users= businesswise_users.collectAsMap()
    userwise_businesses = train_rdd.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x + y).collectAsMap()
    businesswise_users = train_rdd.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y)
    businesswise_ratings = train_rdd.map(lambda x: (x[1], [x[2]])).reduceByKey(lambda x, y: x + y)
    normalized = businesswise_ratings.map(lambda x: get_normalised_ratings(x))

    businesswise_user_ratings = businesswise_users.join(normalized).map(
        lambda x: ((x[0], x[1][0]), (x[1][1][0], x[1][1][1])))
    keyed_ratings = businesswise_user_ratings.flatMap(lambda x: convert2(x)).collectAsMap()
    list_businesseswise_user_ratings = businesswise_user_ratings.map(
        lambda x: (x[0][0], (x[0][1], x[1][0], x[1][1]))).collectAsMap()

    test_pairs = test_rdd.map(lambda x: (x[0], x[1]))

    predicted_ratings = test_pairs.map(
        lambda x: get_item_rating(x, userwise_businesses, list_businesseswise_user_ratings, keyed_ratings))
    # for i in predicted_ratings.collect():
    # 	print(str(i))

    # ratesAndPreds = test_rdd.map(lambda x: ((x[0], x[1]), float(x[2]))).join(predicted_ratings)
    # MSE = ratesAndPreds.map(lambda r: ((r[1][0] - r[1][1])*(r[1][0] - r[1][1]))).mean()
    # print("Root Mean Squared Error = " + str(math.sqrt(MSE)))

    # ====================================================== WRITING TO OUTPUT FILE ======================================================
    f = open(out_file, 'w')

    f.write("user_id, business_id, prediction")
    for i in predicted_ratings.collect():
        f.write("\n")
        f.write(i[0][0] + "," + i[0][1] + "," + str(i[1]))
    f.close()


# ====================================================== MAIN DRIVER PROGRAM =========================================================
train_file = sys.argv[1]
test_file = sys.argv[2]
out_file = sys.argv[3]

sc = SparkContext(appName="PythonCollaborativeFilteringExample")

# Load and parse the data
train_data = sc.textFile(train_file)
train_data = train_data.map(lambda x: x.split(','))
train_rdd = train_data.filter(lambda x: x[0] != "user_id").persist()

test_data = sc.textFile(test_file)
test_data = test_data.map(lambda x: x.split(','))
test_rdd = test_data.filter(lambda x: x[0] != "user_id").persist()

N = 14
N2 = 4000
N3 = 4000
itembased_cf_recommendation(train_rdd, test_rdd)
# ----------------------------------------------- PROCESSING TRAINING DATA -----------------------------------------------
end = time.time()
print("Duration: " + str(end - start))