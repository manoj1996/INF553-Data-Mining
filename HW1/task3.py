import sys
from pyspark import SparkContext
import json
import time

sc = SparkContext(master="local[*]", appName="join2files")
sc.setLogLevel("ERROR")
review_path = sys.argv[1]
business_path = sys.argv[2]
output_path_a = sys.argv[3]
output_path_b = sys.argv[4]

reviews_sc = sc.textFile(review_path)
review_rdd = reviews_sc.map(json.loads)\
    .map(lambda x:(x['business_id'], x['stars']))\
    .persist()
business_sc = sc.textFile(business_path)
business_rdd = business_sc.map(json.loads)\
    .map(lambda x:(x['business_id'], x['city']))\
    .persist()
star_city_agg_rdd = business_rdd.join(review_rdd)\
    .map(lambda a: a[1])\
    .mapValues(lambda x: (x,1))\
	.reduceByKey(lambda m,n:(m[0]+n[0], m[1]+n[1]))\
	.mapValues(lambda x: x[0]/x[1])\
	.persist()

task_start_a= time.time()
star_city_avg_list = star_city_agg_rdd.collect()
result_method_a= sorted(star_city_avg_list, key=lambda x:(-x[1],x[0]))

print("\nPrinting output of first method:\n")
for city, stars in result_method_a[:10]:
	print(city, ",", stars)

task_start_b= time.time()

result_method_b= star_city_agg_rdd.sortBy(lambda x: (-x[1], x[0]), ascending=True).collect()[:10]
print("\nPrinting output of second method:\n")
for city, stars in result_method_b:
	print(city, ",", stars)

task_end_b= time.time()

with open(output_path_a, 'w') as filea:
	filea.write("city,stars\n")
	for x in result_method_a[:len(result_method_a)-1] :
		filea.write(str(x[0])+","+str(x[1]) + "\n")
	filea.write(str(result_method_a[-1][0]) + "," + str(result_method_a[-1][1]))
	filea.close()


with open(output_path_b, 'w') as fileb:
	fileb.write("{\n\t\"m1\":"+str(task_start_b-task_start_a)+",")
	fileb.write("\n\t\"m2\":"+str(task_end_b-task_start_b)+"\n}")
	fileb.close()
