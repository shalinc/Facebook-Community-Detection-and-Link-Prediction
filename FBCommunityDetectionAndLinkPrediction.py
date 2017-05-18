# coding: utf-8

#
# In this assignment, we'll implement community detection and link prediction algorithms using Facebook "like" data.
# The file `edges.txt.gz` indicates like relationships between facebook users. This was collected using snowball sampling: beginning with the user "Bill Gates", I crawled all the people he "likes", then, for each newly discovered user, I crawled all the people they liked.
# We'll cluster the resulting graph into communities, as well as recommend friends for Bill Gates.
#

from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request

## Community Detection

def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.
    
    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.
    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    
    """
    ###TODO
    node2distance = defaultdict(int)
    node2parent = defaultdict(list)
    node2num_path = defaultdict(int)

    visted_nodes = []
    queue = deque()

    #initially for root node
    queue.append(root)
    visted_nodes.append(root)
    #print("Nodes in QUEUE", queue)

    #finding the shortest path node2distance
    while queue:
        #pop from queue the node and visit its neighbors
        #print("Nodes in QUEUE", queue)
        parent = queue.popleft()
        node_neighbors = sorted(graph.neighbors(parent))

        for neighbor in node_neighbors:
            #finding the shortest distances
            if not neighbor in visted_nodes:
                if (node2distance[parent]) < max_depth:
                    #mark the neighbor visted and add in queue
                    queue.append(neighbor)
                    visted_nodes.append(neighbor)
                    #print("Nodes in QUEUE", queue)
                    #Calculate the distance
                    node2distance[neighbor] = node2distance[parent] + 1

            #finding the parents for the nodes node2parent
            if (node2distance[parent]) < max_depth:
                if node2distance[neighbor] == node2distance[parent] + 1:
                        #store the path
                        node2parent[neighbor].append(parent)
                        node2num_path[neighbor] += 1


    node2num_path[root] = 1

    # for node, depth in node2distance.items():
    #     if depth > 0:
    #         for node_parent, depth_parent in node2distance.items():
    #             if depth_parent == (depth - 1) and node in graph.neighbors(node_parent):
    #                 node2parent[node].append(node_parent)

    #finding the number of shortest path from root to nodes

    # for node, parent in node2parent.items():
    #     node2num_path[node] = len(parent)

    #print(sorted(node2distance.items()))
    #print(sorted(node2parent.items()))
    #print(sorted(node2num_path.items()))

    return node2distance, node2num_path, node2parent


def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.
    
    The third and final step is to calculate for each edge e the sum
    over all nodes Y of the fraction of shortest paths from the root
    X to Y that go through e.
    
	Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')).
    """
    ###TODO
    temp_dict = defaultdict(int)
    result = defaultdict(int)

    for node, paths in sorted(node2num_paths.items()):
        temp_dict[node] = 1/paths

    #print(temp_dict.items())

    #print(sorted(node2parents.items(),key=lambda x: x[1]))

    for node, _ in sorted(node2distances.items(),key=lambda x: -x[1]):
        parents = node2parents[node]
        for parent in parents:
            temp_dict[parent] += temp_dict[node]
            #print("Parent =",parent,temp_dict[parent], " Child =",children, temp_dict[children])
            #temp_parent,temp_children = parent, children
            #sort_val = sorted([temp_parent,temp_children])
            #result[(sort_val[0], sort_val[1])] = temp_dict[children]
            #temp_val = temp_dict[node]
            result[tuple(sorted([parent, node]))] = temp_dict[node]

    #print(sorted(temp_dict.items()))
    #print(sorted(result.items()))
    return result



def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.
    
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.
    Returns:
      A dict mapping edges to betweenness.    
	"""
    ###TODO
    approx_betweenness = defaultdict(int)
    for node in graph.nodes():
        #print(node)
        node2distances, node2num_paths, node2parents = bfs(graph, node, max_depth)
        result = bottom_up(node, node2distances, node2num_paths, node2parents)

        for pair, betweenness in result.items():
            approx_betweenness[pair] += betweenness/2

    #print(sorted(approx_betweenness.items()))
    return approx_betweenness


def partition_girvan_newman(graph, max_depth):
    """
    Used approximate_betweenness implementation to partition a graph.
    Computed the approximate betweenness of all edges, and removed
    them until multiple comonents are created.
    
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.
    Returns:
      A list of networkx Graph objects, one per partition.

    """
    ###TODO
    approx_betweeness = approximate_betweenness(graph,max_depth)
    components = [val for val in nx.connected_component_subgraphs(graph)]
    #print(list(components[0]))
    #print(approx_betweeness)
    #print("Components: ", len(components))
    count_remove = 0
    updatedGraph = graph.copy()
    while len(components) == 1:
        #print(list(component) for component in components)
        edge_to_remove = sorted(approx_betweeness.items(),key= lambda x: (-x[1],x[0]))[count_remove][0]
        #print(edge_to_remove)
        #print("Edges: ",graph.number_of_edges())
        updatedGraph.remove_edge(*edge_to_remove)
        count_remove+=1
        #print("Edges: ",graph.number_of_edges())
        components = [val for val in nx.connected_component_subgraphs(updatedGraph)]
        #print(list(components[0]))

    # for component in components:
    #     print(list(component))
    #     #print(val)
    return components


def get_subgraph(graph, min_degree):
    """
	Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.
    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.
    
    """
    ###TODO

    """Intially considering the entire graph as Subgraph,
    then removing the nodes which don't satisfy min_degree criteria
    """
    subgraph = graph.copy()

    for node, degree in graph.degree().items():
        if degree < min_degree:
            subgraph.remove_node(node)

    return subgraph



""""
Compute the normalized cut for each discovered cluster.
I've broken this down into the three next methods.
"""

def volume(nodes, graph):
    """
    Compute the volume for a list of nodes, which
    is the number of edges in `graph` with at least one end in
    nodes.
    Params:
      nodes...a list of strings for the nodes to compute the volume of.
      graph...a networkx graph
    
    """
    ###TODO

    volume_calc = 0

    for edge1, edge2 in graph.edges():
        if (edge1 in nodes) or (edge2 in nodes):
            volume_calc+=1

    return volume_calc


def cut(S, T, graph):
    """
    Compute the cut-set of the cut (S,T), which is
    the set of edges that have one endpoint in S and
    the other in T.
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An int representing the cut-set.
    
    """
    ###TODO


    cut_set = 0
    for edge1, edge2 in graph.edges():
        if ((edge1 in S) and (edge2 in T)) or ((edge1 in T) and (edge2 in S)):
            cut_set += 1

    return cut_set



def norm_cut(S, T, graph):
    """
    The normalized cut value for the cut S/T.
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An float representing the normalized cut value
    """
    ###TODO
    #print("Volume",volume(S,graph),"Volume 2", volume(T,graph), "Cut",cut(S,T,graph),sep="\n")

    norm_cut_calc = (cut(S, T,graph)/volume(S,graph)) + (cut(S,T,graph)/volume(T,graph))

    return norm_cut_calc


def score_max_depths(graph, max_depths):
    """
    In order to assess the quality of the approximate partitioning method
    we've developed, we will run it with different values for max_depth
    and see how it affects the norm_cut score of the resulting partitions.
    Recall that smaller norm_cut scores correspond to better partitions.
    Params:
      graph........a networkx Graph
      max_depths...a list of ints for the max_depth values to be passed
                   to calls to partition_girvan_newman
    Returns:
      A list of (int, float) tuples representing the max_depth and the
      norm_cut value obtained by the partitions returned by
      partition_girvan_newman. See Log.txt for an example.
    """
    ###TODO
    norm_cuts = []

    # plt.figure()
    # nx.draw_networkx(graph)
    # plt.show()
    #original_graph = graph.copy()

    for max_depth in max_depths:
        components = partition_girvan_newman(graph, max_depth)
        #print(len(components[0]), len(components[1]))
        original_graph = graph.copy()
        norm_cuts.append((max_depth,float(norm_cut(components[0],components[1],original_graph))))

    return norm_cuts


## Link prediction

# Next, we'll consider the link prediction problem. In particular,
# we will remove 5 of the accounts that Bill Gates likes and
# compute our accuracy at recovering those links.

def make_training_graph(graph, test_node, n):
    """
    To make a training graph, we need to remove n edges from the graph.
    we'll assume there is a test_node for which we will
    remove some edges. Remove the edges to the first n neighbors of
    test_node, where the neighbors are sorted alphabetically.
    
    Params:
      graph.......a networkx Graph
      test_node...a string representing one node in the graph whose
                  edges will be removed.
      n...........the number of edges to remove.
    Returns:
      A *new* networkx Graph with n edges removed.
    
    """
    ###TODO
    newGraph = graph.copy()
    neighbors = sorted(newGraph.neighbors(test_node))[:n]

    for neighbor in neighbors:
        newGraph.remove_edge(test_node,neighbor)

    return newGraph




def jaccard(graph, node, k):
    """
    Compute the k highest scoring edges to add to this node based on
    the Jaccard similarity measure.
 
    Params:
      graph....a networkx graph
      node.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.
    Returns:
      A list of tuples in descending order of score representing the
      recommended new edges. Ties are broken by
      alphabetical order of the terminal node in the edge.
    """
    ###TODO

    #train_graph = make_training_graph(graph,node,k)
    all_neighbors = set(graph.neighbors(node))
    scores = []

    for n_node in graph.nodes():
        #no edge exists and node is not similar to itself
        if not graph.has_edge(node, n_node) and node!= n_node:
            node_neighbors = set(graph.neighbors(n_node))
            scores.append(((str(node),str(n_node)), 1. * len(all_neighbors & node_neighbors) / len(all_neighbors | node_neighbors)))

    return sorted(scores, key=lambda x: (-x[1],x[0][1]))[0:k]


# One limitation of Jaccard is that it only has non-zero values for nodes two hops away.
#
# Implement a new link prediction function that computes the similarity between two nodes $x$ and $y$  as follows:
#
# $$
# s(x,y) = \beta^i n_{x,y,i}
# $$
#
# where
# - $\beta \in [0,1]$ is a user-provided parameter
# - $i$ is the length of the shortest path from $x$ to $y$
# - $n_{x,y,i}$ is the number of shortest paths between $x$ and $y$ with length $i$


def path_score(graph, root, k, beta):
    """
    Compute a new link prediction scoring function based on the shortest
    paths between two nodes.
    
    Params:
      graph....a networkx graph
      root.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.
      beta.....the beta parameter in the equation above.
    Returns:
      A list of tuples in descending order of score. Ties are broken by
      alphabetical order of the terminal node in the edge.
    
    """
    ###TODO
    node2distances, node2num_paths, node2parents = bfs(graph, root, math.inf)

    return sorted([((root,node),(beta**node2distances[node]) * node2num_paths[node])
                   for node in graph.nodes() if not graph.has_edge(node, root) and node!= root], key=lambda x: (-x[1],x[0][1]))[:k]


def evaluate(predicted_edges, graph):
    """
    Return the fraction of the predicted edges that exist in the graph.
    Args:
      predicted_edges...a list of edges (tuples) that are predicted to
                        exist in this graph
      graph.............a networkx Graph
    Returns:
      The fraction of edges in predicted_edges that exist in the graph.
    
    """
    ###TODO
    predict_right = 0
    for predicted_edge in predicted_edges:
        if graph.has_edge(*predicted_edge):
            predict_right+=1

    return predict_right/len(predicted_edges)




def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():
    
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    
	subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))

    #MYCALLs
    #a,b,c=bfs(example_graph(),'E',5)
    #bottom_up('E',a,b,c)
    #approximate_betweenness(example_graph(), 2)
    #partition_girvan_newman(example_graph(), 5)
    #volume(['A','B','C'], example_graph())
    #norm_cut(['A', 'B', 'C'], ['D', 'E', 'F', 'G'],example_graph())

    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,5)))

    clusters = partition_girvan_newman(subgraph, 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
           (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())

    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))

    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
          evaluate([x[0] for x in jaccard_scores], subgraph))

    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' %
          evaluate([x[0] for x in path_scores], subgraph))


if __name__ == '__main__':
    main()