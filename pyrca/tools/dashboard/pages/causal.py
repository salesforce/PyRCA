default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(id)',
            'opacity': 'data(weight)',
            'background-color': 'blue',
        }
    },
    {
        'selector': 'edge',
        'style': {
            'curve-style': 'bezier',
            'target-arrow-color': 'black',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 1,
            'line-color': 'black',
            'opacity': 0.3,
            'width': 0.5
        }
    },
]


def build_cyto_graph(graph):
    cy_edges = []
    cy_nodes = []
    for node in graph.nodes():
        cy_nodes.append({"data": {"id": node, "label": node}})
    for edge in graph.edges():
        cy_edges.append({"data": {"source": edge[0], "target": edge[1]}})
    return cy_nodes + cy_edges
