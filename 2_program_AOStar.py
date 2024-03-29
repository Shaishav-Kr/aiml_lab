from heuristicsearch.ao_star import AOStar

print('Graph-1')
heuristic = {
    'A': 1,
    'B': 6,
    'C': 2,
    'D': 12,
    'E': 2,
    'F': 1,
    'G': 5,
    'H': 7,
    'I': 7,
    'J': 1
}
graph_nodes = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('G', 1)], [('H', 1)]],
    'C': [[('J', 1)]],
    'D': [[('E', 1), ('F', 1)]],
    'G': [[('I', 1)]]
}
graph = AOStar(graph_nodes, heuristic, 'A')
graph.applyAOStar()
