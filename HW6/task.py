import sys
from sklearn.cluster import KMeans
import numpy as np
import time
from sklearn.metrics.cluster import normalized_mutual_info_score

start = time.time()
inputFile, numberOfClusters, outputFile = sys.argv[1], int(sys.argv[2]), sys.argv[3]

class BFR:
    def merge_clusters(self, key1, key2, statistics):
        x = 0
        while(x < d):
            i = x
            statistics[key1][3][i], x, statistics[key1][2][i] = statistics[key1][3][i] + statistics[key2][3][i], x+1, statistics[key1][2][i] + statistics[key2][2][i]
        statistics[key1][1] = statistics[key1][1] + statistics[key2][1]
        statistics[key1][4], statistics[key1][5] = np.sqrt((statistics[key1][3][:] / statistics[key1][1]) - (
                    np.square(statistics[key1][2][:]) / (statistics[key1][1] ** 2))), statistics[key1][2] / statistics[key1][1]
        statistics[key1][0].extend(statistics[key2][0])

    def update_statistics(self, pointid, newpoint, cluster_key, statistics):
        x = 0
        while(x < d):
            i = x
            statistics[cluster_key][3][i], x, statistics[cluster_key][2][i] = statistics[cluster_key][3][i] + newpoint[i] ** 2, x+1, statistics[cluster_key][2][i] + newpoint[i]
        statistics[cluster_key][1] = statistics[cluster_key][1] + 1
        statistics[cluster_key][4], statistics[cluster_key][5] = np.sqrt((statistics[cluster_key][3][:] / statistics[cluster_key][1]) - (
                    np.square(statistics[cluster_key][2][:]) / (statistics[cluster_key][1] ** 2))), statistics[cluster_key][2] / statistics[cluster_key][1]
        statistics[cluster_key][0].append(pointid)

def writeRoundResults(op):
    with open(outputFile, "w") as fw:
        fw.write(op)
    fw.close()

bfr = BFR()
file = open(inputFile, "r")
data = np.array(file.readlines())
file.close()

initial_cluster = {}
temp_data = data
for i in data:
    t = i.replace("\n", "").split(",")
    initial_cluster[int(t[0])] = int(t[1])

lines = int(len(data) * 0.2)

initial_sample = np.random.choice(a=data, size=lines, replace=False)
data = np.setdiff1d(data, initial_sample)
point_ids, pointid_to_point, point_to_pointid = dict(), dict(), dict()

initial_data = list()
ctr, DS_ctr = 0, 0

for i in initial_sample:
    t = i.replace("\n", "").split(",")
    initial_data.append(t[2:])
    point_ids[DS_ctr], pointid_to_point[t[0]], point_to_pointid[str(t[2:])] = t[0], t[2:], t[0]
    ctr, DS_ctr = ctr + 1, DS_ctr + 1
d = len(initial_data[0])
threshold_dist = 2 * (d**0.5)
kmeans = KMeans(n_clusters=10 * numberOfClusters, random_state=0)
clusters = dict()
X = np.array(initial_data)
clusters1 = kmeans.fit_predict(X)
cluster_centers = kmeans.cluster_centers_

ctr = 0
for clusterid in clusters1:
    point = initial_data[ctr]
    ctr += 1
    if (clusterid not in clusters):
        clusters[clusterid] = [point]
    else:
        clusters[clusterid].append(point)

DS, CS, RS = list(), dict(), dict()

for key in clusters.keys():
    if (len(clusters[key]) != 1):
        continue
    pos = initial_data.index(clusters[key][0])
    RS[point_ids[pos]], i = clusters[key][0], pos
    while(i < len(point_ids)-1):
        point_ids[i] = point_ids[i + 1]
        i += 1
    initial_data.remove(clusters[key][0])

X = np.array(initial_data)
kmeans = KMeans(n_clusters=numberOfClusters, random_state=0)
clusters2 = kmeans.fit_predict(X)
cluster_centers = kmeans.cluster_centers_
clusters, DS_statistics = dict(), dict()
ctr = 0
for clusterid in clusters2:
    point = initial_data[ctr]
    if (clusterid not in clusters):
        clusters[clusterid] = [ctr]
    else:
        clusters[clusterid].append(ctr)
    ctr += 1

for key in clusters.keys():
    DS_statistics[key], DS_statistics[key][0] = dict(), list()
    for i in range(len(clusters[key])):
        DS_statistics[key][0].append(point_ids[clusters[key][i]])
    DS_statistics[key][2], DS_statistics[key][1] = np.sum(X[clusters[key], :].astype(np.float), axis=0), len(DS_statistics[key][0])
    DS_statistics[key][5] = np.divide(DS_statistics[key][2], DS_statistics[key][1])
    DS_statistics[key][3] = np.sum((X[clusters[key], :].astype(np.float)) ** 2, axis=0)
    DS_statistics[key][4] = np.sqrt((DS_statistics[key][3][:] / DS_statistics[key][1]) - (np.square(DS_statistics[key][2][:]) / (DS_statistics[key][1] ** 2)))

RS_points = list()
rs_clusters = dict()
for key in RS.keys():
    RS_points.append(RS[key])

kmeans = KMeans(n_clusters=int(len(RS_points) / 2 + 1), random_state=0)
X = np.array(RS_points)
clusters3 = kmeans.fit_predict(X)

ctr = 0
for clusterid in clusters3:
    if (clusterid not in rs_clusters):
        rs_clusters[clusterid] = [ctr]
    else:
        rs_clusters[clusterid].append(ctr)
    ctr += 1

CS_statistics = dict()
for key in rs_clusters.keys():
    if (len(rs_clusters[key]) <= 1):
        continue
    CS_statistics[key], CS_statistics[key][0] = dict(), list()
    for i in rs_clusters[key]:
        CS_statistics[key][0].append(list(RS.keys())[list(RS.values()).index(RS_points[i])])
    CS_statistics[key][2] = np.sum(X[rs_clusters[key], :].astype(np.float), axis=0)
    CS_statistics[key][1] = len(rs_clusters[key])
    CS_statistics[key][5] = CS_statistics[key][2] / CS_statistics[key][1]
    CS_statistics[key][3] = np.sum((X[rs_clusters[key], :].astype(np.float)) ** 2, axis=0)
    CS_statistics[key][4] = np.sqrt((CS_statistics[key][3][:] / CS_statistics[key][1]) - (np.square(CS_statistics[key][2][:]) / (CS_statistics[key][1] ** 2)))

for key in rs_clusters.keys():
    if (len(rs_clusters[key]) > 1):
        for i in range(len(rs_clusters[key])):
            dict_key_to_remove = list(RS.keys())[list(RS.values()).index(RS_points[rs_clusters[key][i]])]
            del RS[dict_key_to_remove]

RS_points = list()
n_points_DS, n_clusters_CS, n_points_CS, n_points_RS = 0, 0, 0, 0
for key in RS.keys():
    RS_points.append(RS[key])

n_points_RS = len(RS_points)

for key in CS_statistics.keys():
    n_points_CS += len(CS_statistics[key][0])
    n_clusters_CS += 1

for key in DS_statistics.keys():
    n_points_DS += len(DS_statistics[key][0])

op = "The intermediate results:\n"
op += "Round 1: " + str(n_points_DS) + "," + str(n_clusters_CS) + "," + str(n_points_CS) + "," + str(n_points_RS)
ite = 1
while(ite < 5):
    ite += 1
    if (ite < 5):
        sizeD = lines
    else:
        sizeD = len(data)
    next_sample = np.random.choice(a=data, size=sizeD, replace=False)
    data = np.setdiff1d(data, next_sample)
    next_data = list()
    index = DS_ctr
    for i in range(len(next_sample)):
        t = next_sample[i].replace("\n", "").split(",")
        item = t[2:]
        next_data.append(item)
        point_ids[DS_ctr] = t[0]
        DS_ctr += 1
        pointid_to_point[t[0]] = item
        point_to_pointid[str(item)] = t[0]

    X = np.array(next_data)

    ctr = 0
    for i in X:
        mindist, mincluster = threshold_dist, -1
        point, pointid = i.astype(np.float), point_ids[index + ctr]
        for key in DS_statistics.keys():
            stddev, centroid, MD = DS_statistics[key][4].astype(np.float), DS_statistics[key][5].astype(np.float), 0
            dim = 0
            while(dim < d):
                MD += ((point[dim] - centroid[dim]) / stddev[dim]) ** 2
                dim += 1
            MD = np.sqrt(MD)
            if (MD >= mindist):
                continue
            mindist, mincluster = MD, key

        if (mincluster > -1):
            bfr.update_statistics(pointid, point, mincluster, DS_statistics)

        else:
            mindist, mincluster = threshold_dist, -1

            for key in CS_statistics.keys():
                stddev, centroid, MD = CS_statistics[key][4].astype(np.float), CS_statistics[key][5].astype(np.float), 0
                dim = 0
                while(dim < d):
                    MD += ((point[dim] - centroid[dim]) / stddev[dim]) ** 2
                    dim += 1
                MD = np.sqrt(MD)

                if (MD < mindist):
                    mindist, mincluster = MD, key

            if (mincluster > -1):
                bfr.update_statistics(pointid, point, mincluster, CS_statistics)
            else:
                RS[pointid] = list(i)
                RS_points.append(list(i))
        ctr += 1

    X = np.array(RS_points)
    rs_clusters, ctr = dict(), 0
    kmeans = KMeans(n_clusters=int(len(RS_points) / 2 + 1), random_state=0)
    clusters4 = kmeans.fit_predict(X)

    for clusterid in clusters4:
        if (clusterid not in rs_clusters):
            rs_clusters[clusterid] = [ctr]
        else:
            rs_clusters[clusterid].append(ctr)
        ctr += 1

    for key in rs_clusters.keys():
        if (len(rs_clusters[key]) > 1):
            k = 0
            if (key not in CS_statistics.keys()):
                k = key
            else:
                while (k in CS_statistics):
                    k += 1

            CS_statistics[k], CS_statistics[k][0] = dict(), list()
            for i in rs_clusters[key]:
                CS_statistics[k][0].append(list(RS.keys())[list(RS.values()).index(RS_points[i])])
            CS_statistics[k][2] = np.sum(X[rs_clusters[key], :].astype(np.float), axis=0)
            CS_statistics[k][1] = len(rs_clusters[key])
            CS_statistics[k][5] = CS_statistics[k][2] / CS_statistics[k][1]
            CS_statistics[k][3] = np.sum((X[rs_clusters[key], :].astype(np.float)) ** 2, axis=0)
            CS_statistics[k][4] = np.sqrt((CS_statistics[k][3][:] / CS_statistics[k][1]) - (np.square(CS_statistics[k][2][:]) / (CS_statistics[k][1] ** 2)))


    for key in rs_clusters.keys():
        if (len(rs_clusters[key]) <= 1):
            continue
        for i in rs_clusters[key]:
            dict_key_to_remove = point_to_pointid[str(RS_points[i])]
            if (dict_key_to_remove not in RS.keys()):
                continue
            del RS[dict_key_to_remove]

    RS_points = list()
    for key in RS.keys():
        RS_points.append(RS[key])

    list_cs_keys, closest = CS_statistics.keys(), dict()
    for x in list_cs_keys:
        min_MD, min_cluster = threshold_dist, x
        for y in list_cs_keys:
            if (x != y):
                stddev1, stddev2 = CS_statistics[x][4], CS_statistics[y][4]
                centroid1, centroid2 = CS_statistics[x][5], CS_statistics[y][5]
                MD1, MD2 = 0, 0
                dim = 0
                while(dim < d):
                    MD1 += np.divide((centroid1[dim] - centroid2[dim]), stddev2[dim]) ** 2
                    MD2 += np.divide((centroid2[dim] - centroid1[dim]), stddev1[dim]) ** 2
                    dim += 1
                MD1, MD2 = np.sqrt(MD1), np.sqrt(MD2)

                MD = min(MD1, MD2)
                if (MD >= min_MD):
                    continue
                min_MD, min_cluster = MD, y

        closest[x] = min_cluster

    for i in closest.keys():
        if (i != closest[i]):
            if (closest[i] in CS_statistics.keys()):
                if (i in CS_statistics.keys()):
                    bfr.merge_clusters(i, closest[i], CS_statistics)
                    del CS_statistics[closest[i]]

    if (ite == 4):
        closest = dict()
        list_cs_keys, list_ds_keys = CS_statistics.keys(), DS_statistics.keys()

        for x in list_cs_keys:
            min_MD, min_cluster = threshold_dist, -30
            for y in list_ds_keys:
                if (x != y):
                    stddev1, stddev2, MD1, MD2 = CS_statistics[x][4], DS_statistics[y][4], 0, 0
                    centroid1, centroid2 = CS_statistics[x][5], DS_statistics[y][5]
                    dim = 0
                    while(dim < d):
                        MD2 += ((centroid2[dim] - centroid1[dim]) / stddev1[dim]) ** 2
                        MD1 += ((centroid1[dim] - centroid2[dim]) / stddev2[dim]) ** 2
                        dim += 1
                    MD1, MD2 = np.sqrt(MD1), np.sqrt(MD2)
                    MD = min(MD1, MD2)
                    if (MD >= min_MD):
                        continue
                    min_MD, min_cluster = MD, y
            closest[x] = min_cluster

        for i in closest.keys():
            if (closest[i] in DS_statistics.keys()):
                if (i in CS_statistics.keys()):
                    bfr.merge_clusters(closest[i],i, DS_statistics)
                    del CS_statistics[i]

    n_points_DS, n_points_CS, n_points_RS, n_clusters_CS = 0, 0, len(RS_points), 0

    for key in CS_statistics.keys():
        n_points_CS += len(CS_statistics[key][0])
        n_clusters_CS += 1

    for key in DS_statistics.keys():
        n_points_DS += len(DS_statistics[key][0])

    print("Round " + str(ite) + ": " + str(n_points_DS) + "," + str(n_clusters_CS) + "," + str(
        n_points_CS) + "," + str(n_points_RS))
    op += "\nRound " + str(ite) + ": " + str(n_points_DS) + "," + str(n_clusters_CS) + "," + str(
        n_points_CS) + "," + str(n_points_RS)


op += "\n"
cluster_l, clusterlist = dict(), dict()
for key in DS_statistics.keys():
    for point in DS_statistics[key][0]:
        clusterlist[point] = key
        cluster_l[int(point)] = key
for key in CS_statistics:
    for point in CS_statistics[key][0]:
        clusterlist[point] = -1
        cluster_l[int(point)] = -1
for key in RS:
    clusterlist[key] = -1
    cluster_l[int(key)] = -1

op += "\nThe clustering results:"
for key in sorted(clusterlist.keys(), key=int):
    op += "\n" + str(key) + "," + str(clusterlist[key])

writeRoundResults(op)

end = time.time()
print("Duration: " + str(end - start))