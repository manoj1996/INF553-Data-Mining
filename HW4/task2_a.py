from pyspark import SparkContext
import sys
import time
from itertools import combinations
from collections import defaultdict
import queue as Q   # For BFS Implementation #
import copy
sc = SparkContext("local[*]", "Task1")
sc.setLogLevel("OFF")
start = time.time()
input_file_path = sys.argv[1]
RDD_inter = sc.textFile(input_file_path)
result_RDD1 = RDD_inter.map(lambda a: a.split(" "))
result_RDD2 = RDD_inter.map(lambda a: a.split(" ")[::-1])
result_RDD = result_RDD1.union(result_RDD2)

#print(user_pairs_RDD.take(5))
nodes_RDD = result_RDD.flatMap(lambda a: [(a[0]),(a[1])]).distinct()
nodes_list = nodes_RDD.collect()
print(len(nodes_list))
edges_RDD = result_RDD.map(lambda a: (a[0], a[1])).map(lambda a: (a[0], a[1]))
edges_list = edges_RDD.collect()
print(len(edges_list))

# Edges between users based on threshold #
def user_edges(user):
    user_edge_list = []
    for i in edges_list:
        if (i[1] == user):
            user_edge_list.append(i[0])
        elif (i[0] == user):
            user_edge_list.append(i[1])
    user_edge_list = list(set(user_edge_list))
    return user_edge_list
nodes_edges = nodes_RDD.map(lambda user: (user, user_edges(user))).collectAsMap()
#print(nodes_edges)
#
# # Betweenness Calculation #
def betweenness(root, nodes_edges):
    no_of_vertices = len(nodes_list)
    que = Q.Queue(maxsize = no_of_vertices)
    node_visited_list = list()
    node_levels = dict()
    node_parents = dict()
    node_weights = dict()

    que.put(root) # Inserting a node into the queue #
    node_visited_list.append(root) # Adding root to the visited list #
    node_levels[root] = 0 # Setting the root node level to zero #
    node_weights[root] = 1

    while (que.empty() != True):
        node = que.get() # get the node from the queue #
        children = nodes_edges[node]

        for i in children:
            count_vertex = 0
            if (i not in node_visited_list):
                que.put(i)
                node_parents[i] = [node]
                node_weights[i] = node_weights[node]
                node_visited_list.append(i)
                node_levels[i] = node_levels[node] + 1
            else:
                if (i != root):
                    node_parents[i].append(node)
                    if (node_levels[node] == node_levels[i] - 1):
                        node_weights[i] = node_weights[i] + node_weights[node]

    node_order = list()
    count = 0
    for v in node_visited_list:
        node_order.append((v, count))
        count += 1
    node_reverse = sorted(node_order, key=(lambda a: a[1]), reverse=True)
    rev_order = list()
    node_values = dict()
    for i in node_reverse:
        rev_order.append(i[0])
        node_values[i[0]] = 1
    betweenness_value = dict()
    for j in rev_order:
        reverse_count = 0
        if (j != root):
            total_weight = 0
            for i in node_parents[j]:
                if (node_levels[i] == node_levels[j] - 1):
                    total_weight = total_weight + node_weights[i]
            for i in node_parents[j]:
                if (node_levels[i] == node_levels[j] - 1):
                    pair_values = 0
                    node_1 = j
                    node_2 = i
                    if node_1 < node_2:
                        node_pair = tuple((node_1, node_2))
                    else:
                        node_pair = tuple((node_2, node_1))
                    val = float(node_values[node_1] * node_weights[node_2] / total_weight)
                    if (node_pair not in betweenness_value):
                        betweenness_value[node_pair] = val
                    else:
                        betweenness_value[node_pair] = betweenness_value[node_pair] + val
                    node_values[node_2] = node_values[node_2] + val
    final_betweenness = list()
    for x,y in betweenness_value.items():
        value = [x,y]
        final_betweenness.append(value)
    #print(final_betweenness)
    return final_betweenness

betweenness_RDD = nodes_RDD.flatMap(lambda a: betweenness(a, nodes_edges)).reduceByKey(lambda a,b: (a+b)).map(lambda a: (a[0],(a[1]/2.0))).sortByKey().map(lambda a: (a[1],a[0])).sortByKey(False).map(lambda a: (a[1],a[0]))
opened_file = open(sys.argv[2], 'w')
for val in betweenness_RDD.collect():
    opened_file.write(str(val[0])+", "+str(val[1])+ "\n")
opened_file.close()
#
# The logic is to remove the edge with highest betweenness in each iteration and get the connected components of the graph using BFS #
def BFS(root, adjacent_vertices, no_vertices):
    nodes_visited = []
    node_edges = []
    que = Q.Queue(maxsize=no_vertices)
    que.put(root)
    nodes_visited.append(root)
    while (que.empty() != True):
        node = que.get()
        children = adjacent_vertices[node]
        for c in children:
            if (c not in nodes_visited):
                que.put(c)
                nodes_visited.append(c)
            node_pair = sorted((node, c))
            if (node_pair not in node_edges):
                node_edges.append(node_pair)
    return (nodes_visited, node_edges)


def check_empty(adjacent_vertices):
    node_count = len(adjacent_vertices)
    if node_count == 0:
        return True
    else:
        for v in adjacent_vertices:
            empty_node_list = list()
            node_list = adjacent_vertices[v]
            if (len(node_list) != 0):
                return False
            else:
                pass
        return True

new_graph_RDD = nodes_RDD.count()
#print(new_graph_RDD)

# Removing the Components from the graph #
def removing_component(remaining_data, component):
    c_vertices = component[0]
    for v in c_vertices:
        del remaining_data[v]
    for k in remaining_data:
        remaining_node_list = list()
        node_list = remaining_data[k]
        for v in c_vertices:
            if (v in node_list):
                node_list.remove(k[1])
        remaining_data[k] = node_list
    return remaining_data

# Connected Component Calculation #
def connected_components_calculation(adjacent_vertices):
    connected_components = []
    graph_data = adjacent_vertices
    while (check_empty(graph_data) == False):
        nodes = list()
        for k,v in graph_data.items():
            nodes.append(k)
        nodes = list(set(nodes))
        root_node = nodes[0]
        res = BFS(root_node, adjacent_vertices, len(nodes))
        connected_components.append(res)
        graph_data = removing_component(graph_data, res)
    return connected_components

# Modularity Calculation #
def modularity_calculation(adjacent_vertices, connected_components, m):
    modularity = 0
    for c in connected_components:
        c_vertices = c[0]
        for i in c_vertices:
            for j in c_vertices:
                Aij = 0
                node_list = adjacent_vertices[str(i)]
                if (j in node_list):
                    Aij = 1
                ki = len(adjacent_vertices[i])
                kj = len(adjacent_vertices[j])
                val = ki * kj
                modularity += Aij - val / (2 * m)
    modularity = modularity / (2 * m)
    return modularity

# Adjacency Matrix Construction #
def adjacency_matrix_construction(connected_components):
    adj_matrix = dict()
    for c in connected_components:
        c_edges = c[1]
        for i in c_edges:
            if (i[0] in adj_matrix):
                adj_matrix[i[0]].append(i[1])
            else:
                adj_matrix[i[0]] = [i[1]]
            if (i[1] in adj_matrix):
                adj_matrix[i[1]].append(i[0])
            else:
                adj_matrix[i[1]] = [i[0]]
    return adj_matrix

# Removing the highest betweenness edge #
def edge_removal(adjacency_matrix, edge):
    if (edge[0] in adjacency_matrix):
        val = adjacency_matrix[edge[0]]
        if (edge[1] in val):
            val.remove(edge[1])
    if (edge[1] in adjacency_matrix):
        val = adjacency_matrix[edge[1]]
        if (edge[0] in val):
            val.remove(edge[0])
    return adjacency_matrix

edge_to_remove = betweenness_RDD.take(1)[0][0]
m = edges_RDD.count()
adjacency_matrix = nodes_edges.copy()
highest_modularity = -1
communities_list = list()
count_iter = 0
try:
    while(True):
        adjacency_matrix = edge_removal(adjacency_matrix, edge_to_remove)
        connected_components = connected_components_calculation(adjacency_matrix)
        modularity = modularity_calculation(nodes_edges, connected_components, m)
        adjacency_matrix = adjacency_matrix_construction(connected_components)
        temp_list = list()
        for i in adjacency_matrix:
            temp_list.append(i)
        temp_list = list(set(temp_list))
        v_RDD = sc.parallelize(temp_list)
        temp_betweenness = v_RDD.flatMap(lambda x: betweenness(x, adjacency_matrix)).reduceByKey(lambda a,b: (a+b)).map(lambda a: (a[0], (a[1]/2.0))).sortByKey().map(lambda a: (a[1],a[0])).sortByKey(ascending=False).map(lambda a: (a[1],a[0]))
        edge_to_remove= temp_betweenness.take(1)[0][0]
        if(modularity >= highest_modularity):
            highest_modularity = modularity
            communities_list = connected_components
        count_iter = count_iter + 1
        if(count_iter == 650):
            break
except:
    print("Converged")
#print(communities_list)
communities_sorted = list()
for i in communities_list :
    value = sorted(i[0])
    communities_sorted.append((value,len(value)))
community_count = 0
communities_sorted.sort()
communities_sorted.sort(key=lambda a:a[1])
opened_file = open(sys.argv[3], "w")
for i in communities_sorted:
    val = str(i[0]).replace("[","").replace("]","")
    opened_file.write(val)
    opened_file.write("\n")
opened_file.close()
end = time.time()
print("Duration: ", end - start)
