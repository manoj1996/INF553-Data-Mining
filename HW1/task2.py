import sys
from pyspark import SparkContext
import json
import time
from operator import add

start_time= time.time()
sc = SparkContext(master="local[*]", appName="partitioning")
sc.setLogLevel("ERROR")
business_file = sys.argv[1]
output_file = sys.argv[2]
num_partitions = int(sys.argv[3])
output = []
output.append(str("{\n\t\"default\":{"))
business_text_file = sc.textFile(business_file)
business_rdd=business_text_file.map(json.loads)\
    .map(lambda x:(x['business_id'],1)).persist()

def item_per_partition(num, partition):
    return num, sum([1 for _ in partition])
def custom_partition(business_id):
    return hash(business_id)

task1_start_time= time.time()
myList1= business_rdd.mapPartitionsWithIndex(item_per_partition).collect()
list1= myList1[1::2]

task_start1= time.time()
business_list1_sorted= business_rdd.map(lambda row:(row[0],1))\
    .reduceByKey(add)\
    .sortBy(lambda x: (-x[1], x[0]), ascending=True)
task_end1= time.time()
output.append(str("\n\t\t\"n_partition\":"+str(business_rdd.getNumPartitions())+","))
output.append(str("\n\t\t\"n_items\":"+str(list1)+","))
output.append(str("\n\t\t\"exe_time\":"+str(task_end1-task_start1)+"\n\t},"))
output.append(str("\n\t\"customized\":{"))
task_part2_start= time.time()

business_rdd2= business_rdd.partitionBy(num_partitions, custom_partition)
myList2= business_rdd2.mapPartitionsWithIndex(item_per_partition).collect()
list2= myList2[1::2]

task_start2= time.time()
business_list2_sorted= business_rdd2.map(lambda row:(row[0],1))\
    .reduceByKey(add)\
    .sortBy(lambda x: (-x[1], x[0]), ascending=True)
task_end2= time.time()
output.append(str("\n\t\t\"n_partition\":"+str(business_rdd2.getNumPartitions())+","))
output.append(str("\n\t\t\"n_items\":"+str(list2)+","))
output.append(str("\n\t\t\"exe_time\":"+str(task_end2-task_start2)+"\n\t}\n}"))
ts_write_start= time.time()
with open(output_file, 'w') as fw:
    for out in output:
        fw.write(out)
    fw.close()
end_time = time.time()
print("Program execution:", end_time-start_time)