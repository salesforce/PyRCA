from dash import Dash, html, Input, Output, State, dcc, ctx
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from dash.dependencies import Input, Output, State
import json

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

cyto.load_extra_layouts()

app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape',
        elements=[
            {'data': {'id': 'one', 'label': 'Node 1'},
             'position': {'x': 0, 'y': 1}},
            {'data': {'id': 'two', 'label': 'Node 2'},
             'position': {'x': 1, 'y': 2}},
            {'data': {'id': 'three', 'label': 'Node 3'},
             'position': {'x': 2, 'y': 4}},
            {'data': {'id': 'four', 'label': 'Node 4'},
             'position': {'x': 5, 'y': 6}},
            {'data': {'source': 'one', 'target': 'two', 'label': '1 to 2'}},
            {'data': {'source': 'two', 'target': 'three', 'label': '2 to 3'}},
            {'data': {'source': 'three', 'target': 'one', 'label': '1 to 3'}},
            {'data': {'source': 'three', 'target': 'four', 'label': '3 to 4'}}
        ],
        # layout={'name': 'cose'}
        layout={'name': 'breadthfirst'}
    ),
    html.Button("Print elements JSONified", id="button-cytoscape"),
    html.Div(id="html-cytoscape"),
])


@app.callback(
    Output("html-cytoscape", "children"),
    [Input("button-cytoscape", "n_clicks")],
    [State("cytoscape", "elements")],
)
def testCytoscape(n_clicks, elements):
    if n_clicks:
        return json.dumps(elements)


if __name__ == '__main__':
    app.run_server(debug=True)
