#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause#
import json
import dash
from dash import Input, Output, State, callback
from ..pages.data import create_stats_table, create_metric_stats_table
from ..utils.file_manager import FileManager
from ..models.data import DataAnalyzer

file_manager = FileManager()
data_analyzer = DataAnalyzer(folder=file_manager.data_directory)


@callback(
    Output("select-file", "options"),
    Output("select-file", "value"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def upload_file(filenames, contents):
    name = None
    if filenames is not None and contents is not None:
        for name, data in zip(filenames, contents):
            file_manager.save_file(name, data)
    options = []
    files = file_manager.uploaded_files()
    for filename in files:
        options.append({"label": filename, "value": filename})
    return options, name


@callback(
    Output("data-stats-table", "children"),
    Output("data-state", "data"),
    Output("data-table", "children"),
    Output("data-plots", "children"),
    Output("data-exception-modal", "is_open"),
    Output("data-exception-modal-content", "children"),
    [
        Input("data-btn", "n_clicks"),
        Input("thres-btn", "n_clicks"),
        Input("manual-btn", "n_clicks"),
        Input("data-exception-modal-close", "n_clicks"),
    ],
    [
        State("select-file", "value"),
        State("select-thres-column", "value"),
        State("sigma", "value"),
        State("lower_bound", "value"),
        State("upper_bound", "value"),
        State("data-state", "data"),
    ],
)
def click_run(
    btn_click, thres_click, manual_click, modal_close, file_name, thres_column, sigma, lower_bound, upper_bound, data
):
    ctx = dash.callback_context
    stats = json.loads(data) if data is not None else {}

    stats_table = create_stats_table()
    data_table = DataAnalyzer.get_data_table(df=None)
    data_figure = DataAnalyzer.get_data_figure(df=None)
    modal_is_open = False
    modal_content = ""

    try:
        if ctx.triggered:
            prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if prop_id == "data-btn" and btn_click > 0:
                assert file_name, "Please select a file to load."
                df = data_analyzer.load_data(file_name)
                stats = DataAnalyzer.get_stats(df)
                stats_table = create_stats_table(stats)
                data_table = DataAnalyzer.get_data_table(df)
                data_figure = DataAnalyzer.get_data_figure(df)

            elif prop_id == "thres-btn" and thres_click > 0:
                data_figure = data_analyzer.estimate_threshold(column=thres_column, sigma=float(sigma))

            elif prop_id == "manual-btn" and manual_click > 0:
                data_figure = data_analyzer.manual_threshold(
                    column=thres_column, lower=float(lower_bound), upper=float(upper_bound)
                )

    except Exception as error:
        modal_is_open = True
        modal_content = str(error)

    return stats_table, json.dumps(stats), data_table, data_figure, modal_is_open, modal_content


@callback(Output("select-column", "options"), Input("select-column-parent", "n_clicks"), State("data-state", "data"))
def update_metric_dropdown(n_clicks, data):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-column-parent":
            stats = json.loads(data)
            options += [{"label": s, "value": s} for s in stats.keys() if s.find("@") == -1]
    return options


@callback(Output("metric-stats-table", "children"), Input("select-column", "value"), State("data-state", "data"))
def update_metric_table(column, data):
    ctx = dash.callback_context
    metric_stats_table = create_metric_stats_table()

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-column" and data is not None:
            stats = json.loads(data)
            metric_stats_table = create_metric_stats_table(stats, column)
    return metric_stats_table


@callback(
    Output("select-thres-column", "options"),
    Input("select-thres-column-parent", "n_clicks"),
    State("data-state", "data"),
)
def update_thres_dropdown(n_clicks, data):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-thres-column-parent":
            stats = json.loads(data)
            options += [{"label": s, "value": s} for s in stats.keys() if s.find("@") == -1]
    return options


@callback(
    Output("select-manual-column", "options"),
    Input("select-manual-column-parent", "n_clicks"),
    State("data-state", "data"),
)
def update_manual_dropdown(n_clicks, data):
    options = []
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "select-manual-column-parent":
            stats = json.loads(data)
            options += [{"label": s, "value": s} for s in stats.keys() if s.find("@") == -1]
    return options
