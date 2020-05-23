import sys
from pyspark import SparkContext
from operator import add
import time
from queue import *
start = time.time()
sc = SparkContext(master="local[*]", appName="girvan-newman")
sc.setLogLevel("ERROR")
inputFile, betweenessOutput, communityOutput = sys.argv[1], sys.argv[2], sys.argv[3]

class girvanNewman:
    def findEdges(self, x, listEdges):
        edges = list()
        for i in listEdges:
            tmp = None
            if(i[0] == x):  
                tmp = i[1]
            elif(i[1] == x):
                tmp = i[0]
            if tmp is not None:
                edges.append(tmp)
        edges = list(set(edges))
        return edges

    def calculateBetweenness(self, rootNode, adjacentVertices, nodes):
        q = Queue(maxsize = nodes)
        visited, levels, parents, weights = list(), {}, {}, {}
        visited.append(rootNode)
        q.put(rootNode)
        levels[rootNode], weights[rootNode] = 0, 1
        while not q.empty():
            node = q.get()
            children = adjacentVertices[node]
            for i in children:
                if (i in visited):
                    if (i != rootNode):
                        parents[i].append(node)
                        if (levels[i] - 1 == levels[node]):
                            weights[i] += weights[node]
                else:
                    q.put(i)
                    parents[i], weights[i] = [node], weights[node]
                    visited.append(i)
                    levels[i] = levels[node] + 1

        order_v = list()
        count = -1
        for i in visited:
            count = count + 1
            order_v.append((i, count))
        rev_order, nodes_values = list(), {}
        reverse_order = sorted(order_v, key=(lambda x: x[1]), reverse=True)

        for i in reverse_order:
            nodes_values[i[0]] = 1
            rev_order.append(i[0])
        betweennessValues = {}

        for j in range(len(rev_order)):
            if (rev_order[j] != rootNode):
                total_weight = 0
                for i in parents[rev_order[j]]:
                    if (levels[i] == levels[rev_order[j]] - 1):
                        total_weight += weights[i]
                for i in parents[rev_order[j]]:
                    if (levels[i] == levels[rev_order[j]] - 1):
                        source, dest = rev_order[j], i
                        if source >= dest:
                            pair = tuple((dest, source))
                        else:
                            pair = tuple((source, dest))
                        if (pair in betweennessValues.keys()):
                            betweennessValues[pair] += float(nodes_values[source] * weights[dest] / total_weight)
                        else:
                            betweennessValues[pair] = float(nodes_values[source] * weights[dest] / total_weight)
                        nodes_values[dest] += float(nodes_values[source] * weights[dest] / total_weight)

        betweennessList = list()
        for key, value in betweennessValues.items():
            betweennessList.append([key, value])
        return betweennessList

    def remove_edge(self, adjacency_matrix, first_edge_to_remove):
        lhs = first_edge_to_remove[0]
        rhs = first_edge_to_remove[1]
        if lhs in adjacency_matrix:
            l = adjacency_matrix[lhs]
            if rhs in l:
                l.remove(first_edge_to_remove[1])

        if rhs in adjacency_matrix:
            l = adjacency_matrix[rhs]
            if lhs in l:
                l.remove(lhs)
        return adjacency_matrix

    def bfs(self, rootNode, adjacentVertices, nodes):
        visited, edges = list(), list()
        q = Queue(maxsize = nodes)
        q.put(rootNode)
        visited.append(rootNode)
        while not q.empty():
            node = q.get()
            for i in adjacentVertices[node]:
                if (i not in visited):
                    q.put(i)
                    visited.append(i)
                pair = sorted((node, i))
                if (pair in edges):
                    continue
                edges.append(pair)
        return (visited, edges)

    def removeComponent(self, remainderGraph, component):
        component_vertices = component[0]
        for v in range(len(component_vertices)):
            del remainderGraph[component_vertices[v]]

        for key, value in remainderGraph.items():
            adj_list = value
            for v in component_vertices:
                if (v in adj_list):
                    adj_list.remove(key[1])
            remainderGraph[key] = adj_list

        return remainderGraph

    def isEmpty(self, adjacentVertices):
        if (len(adjacentVertices) != 0):
            for i in adjacentVertices.keys():
                adjList = adjacentVertices[i]
                if (len(adjList) == 0):
                    pass
                else:
                    return False
            return True
        else:
            return True


    def getConnectedComponents(self, adjacentVertices):
        connectedComponents = list()
        remainderGraph = adjacentVertices

        while not self.isEmpty(remainderGraph):
            vertices = list()
            for key, value in remainderGraph.items():
                vertices.append(key)
            vertices = list(set(vertices))
            root = vertices[0]
            comp_g = self.bfs(root, adjacentVertices, len(vertices))
            remainderGraph = self.removeComponent(remainderGraph, comp_g)
            connectedComponents.append(comp_g)
        return connectedComponents


    def calculateModularity(self, adjacentVertices, connectedComponents, m):
        modularity = 0
        for c in range(len(connectedComponents)):
            c_vertices = connectedComponents[c][0]
            for i in range(len(c_vertices)):
                for j in range(len(c_vertices)):
                    Aij = 0
                    adj_list = adjacentVertices[str(c_vertices[i])]
                    if (c_vertices[j] in adj_list):
                        Aij = 1
                    kj, ki = len(adjacentVertices[c_vertices[j]]), len(adjacentVertices[c_vertices[i]])
                    temp = (ki * kj) / (2 * m)
                    modularity += Aij - temp
        modularity /= (2*m)
        return modularity

def buildAdjacencyMatrix(connectedComponents):
    res = dict()
    for c in range(len(connectedComponents)):
        c_edges = connectedComponents[c][1]
        for i in c_edges:
            if (i[1] not in res.keys()):
                res[i[1]] = {i[0]}
            else:
                res[i[1]].add(i[0])

            if (i[0] not in res.keys()):
                res[i[0]] = {i[1]}
            else:
                res[i[0]].add(i[1])
    return res

GN = girvanNewman()
rddEdges = sc.textFile(inputFile)\
    .map(lambda x: x.split(' '))\
    .persist()
rdd_edges = rddEdges.map(lambda x: (x[0], x[1])).map(lambda x: (x[0], x[1]))
rdd_vertices = rddEdges.flatMap(lambda x: [(x[0]), (x[1])])\
    .distinct()
list_edges = rdd_edges.collect()
n_vertices = rdd_vertices.count()
adjacent_vertices = rdd_vertices.map(lambda x: (x, GN.findEdges(x, list_edges)))\
    .collectAsMap()
betweenness_rdd = rdd_vertices.flatMap(lambda x: GN.calculateBetweenness(x, adjacent_vertices, n_vertices)) \
    .reduceByKey(add)\
    .map(lambda x: (x[0], float(x[1] / 2)))\
    .sortByKey()\
    .map(lambda x: (x[1], x[0]))\
    .sortByKey(ascending=False)\
    .map(lambda x: (x[1], x[0]))

first_edge_to_remove = betweenness_rdd.take(1)[0][0]
m = len(rdd_edges.collect())
adjacency_matrix = adjacent_vertices.copy()

connected_components = GN.getConnectedComponents(adjacency_matrix)
modularity = GN.calculateModularity(adjacent_vertices, connected_components, m)
adjacency_matrix = adjacent_vertices.copy()
highest_modularity, count = -1, 0
communities = list()
while (len(connected_components) < n_vertices):
    temp = list()
    adjacency_matrix = GN.remove_edge(adjacency_matrix, first_edge_to_remove)
    connected_components = GN.getConnectedComponents(adjacency_matrix)
    modularity = GN.calculateModularity(adjacent_vertices, connected_components, m)
    adjacency_matrix = buildAdjacencyMatrix(connected_components)
    for key, val in adjacency_matrix.items():
        temp.append(key)
    v_rdd = sc.parallelize(list(set(temp)))
    betweenness_temp = v_rdd.flatMap(lambda x: GN.calculateBetweenness(x, adjacency_matrix, n_vertices)) \
        .reduceByKey(add)\
        .map(lambda x: (x[0], float(x[1] / 2)))\
        .sortByKey().map(lambda x: (x[1], x[0]))\
        .sortByKey(ascending=False)\
        .map(lambda x: (x[1], x[0]))
    first_edge_to_remove = betweenness_temp.take(1)[0][0]
    if (modularity >= highest_modularity):
        highest_modularity, communities = modularity, connected_components
    count += 1
    if count >= 150:
        break

sorted_communities = list()
for i in range(len(communities)):
    item = sorted(communities[i][0])
    sorted_communities.append((item, len(item)))
sorted_communities = sorted(sorted_communities)
sorted_communities = sorted(sorted_communities, key=lambda x: x[1])

with open(betweenessOutput, 'w') as fBetween:
    for i in betweenness_rdd.collect():
        fBetween.write(str(i[0]) + ", " + str(i[1]) + "\n")
    fBetween.close()

with open(communityOutput, 'w') as fComm:
    for i in sorted_communities:
        fComm.write(str(i[0]).replace("[", "").replace("]", "") + "\n")
    fComm.close()

end = time.time()
print("Duration: " + str(end - start))