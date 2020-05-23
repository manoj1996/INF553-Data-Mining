import sys
from pyspark import SparkContext
import json
from operator import add
import time

start_time= time.time()
sc = SparkContext(master="local[*]", appName="Review counts")
output = []
sc.setLogLevel("ERROR")
review_file = sys.argv[1]
output_file = sys.argv[2]

review_dataset = sc.textFile(review_file)

def custom_partition(x):
    return hash(str(x[0])+str(x[1])+str(x[2]))
review_rdd=review_dataset.map(json.loads).map(lambda x:((x['business_id'], x['user_id'], x['date']),1)).persist()

review_rdd = review_rdd.partitionBy(20, custom_partition)
output.append(str("{\n\t\"n_review\":"+str(review_rdd.count())+","))
review_2018= review_rdd.map(lambda y:(y[0][2])).filter(lambda x: x.startswith("2018"))
output.append(str("\n\t\"n_review_2018\":"+str(review_2018.count())+","))
users= review_rdd.map(lambda row:(row[0][1]))
output.append(str("\n\t\"n_user\":"+str(users.distinct().count())+","))
users_map= users.map(lambda x:(x,1)).reduceByKey(add)
users_sorted= users_map.sortBy(lambda x: (-x[1], x[0]), ascending=True)
top_10_users= users_sorted.take(10)
top_10_users_str= str(top_10_users).replace('\'','\"').replace('(','[').replace(')',']')
output.append(str("\n\t\"top10_user\":"+top_10_users_str+","))
business= review_rdd.map(lambda row:(row[0][0]))
distinct_business_tran= business.distinct()
output.append(str("\n\t\"n_business\":"+str(distinct_business_tran.count())+","))
business_map= business.map(lambda x:(x,1)).reduceByKey(add)
business_sorted= business_map.sortBy(lambda x: (-x[1], x[0]), ascending=True)
top_10_business= business_sorted.collect()[:10]
top_10_business_str= str(top_10_business).replace('\'','\"').replace('(','[').replace(')',']')
output.append(str("\n\t\"top10_business\":"+top_10_business_str+"\n}"))
end_time = time.time()
print("Time:",end_time-start_time)

with open(output_file, 'w') as fw:
    for out in output:
        fw.write(out)
    fw.close()