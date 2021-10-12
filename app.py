import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output, State, ClientsideFunction, ALL
import plotly.express as px
import pandas as pd
import pdftotext as pt
from flask import Flask, send_file, send_from_directory, redirect
from io import BytesIO
import visdcc
import re
import json
from dash_extensions import Download
from dash_extensions.snippets import send_file
import base64
import lib

server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server = server)

###################################
# Controls
###################################

query = dbc.Card([html.H2("Query"), dcc.Textarea(rows = 5, id = "query")], body = True)

submit = dbc.Button("Submit", id = "submit")
view_document = dbc.Button("View Document", id = "view_document")
upload = dcc.Upload(dbc.Button('Upload File'), id = "upload")

buttons = dbc.Card(dbc.ButtonGroup([upload, view_document, submit]))

threshold = dbc.Card([html.H3("Threshold")])

row1 = html.Tr([html.Td(html.Strong("Parameter 1")), html.Td("1")])
row2 = html.Tr([html.Td(html.Strong("Parameter 2")), html.Td("2")])
parameters_body = [row1, row2]
parameters = html.Table(parameters_body)

settings = dbc.Card([html.H2("Settings"), parameters], body = True)

controls = dbc.Col([query, buttons, settings], width = 4)

###################################
# Summary
###################################

row1 = html.Tr([html.Td(html.Strong("Highlighted words: ")), 
                html.Td(0, 
                        id = "highlighted_words", 
                        style = {"text-align":"right"})])
row2 = html.Tr([html.Td(html.Strong("Highlights: ")), 
                html.Td(0, 
                        id = "highlighted_lines", 
                        style = {"text-align":"right"})])
row3 = html.Tr([html.Td(html.Strong("Reviewed: ")), 
                html.Td(0, 
                        id = "reviews", 
                        style = {"text-align":"right"})])
counts_body = [row1, row2, row3]
counts = dbc.Card(html.Table(counts_body))

summary_title = dbc.CardGroup([dbc.Card(html.H2("Summary")), counts])

view_summary = dbc.Button(
    "View summary",
    id="view_summary",
    n_clicks=0,
)

summary_body = dbc.Card(id = "summary_body",
                        style = {"height":"330px", 
                                 "overflow-y":"scroll", 
                                 "background-color":"lightgrey"})

summary_display = dbc.Col(dbc.Card([summary_title, summary_body, dbc.Card(view_summary)], body = True))

####################################
# View document
###################################

close_document = dbc.Button(
    "Close",
    id="close_document",
    n_clicks=0,
)

document_modal = dbc.Modal(
    [
        dbc.ModalHeader(html.H1("Uploaded document")),
        dbc.ModalBody([dbc.Card(id = "document_modal_body", 
                                style = {"height":"400px", "overflow-y":"scroll"})]),
        dbc.ModalFooter(close_document)
    ],
    id = "document_modal",
    scrollable = False,
    is_open = False,
    size = "xl",
)

##################################
# View summary
###################################

close_summary = dbc.Button(
    "Close",
    id = "close_summary",
    n_clicks = 0,
    style = {"float":"right"}
)

download_summary = dbc.Button(
    "Download",
    id = "download_summary",
    n_clicks=0,
    color = "link"
)

summary_modal = dbc.Modal(
    [
        dbc.ModalHeader(html.H1("Your summary")),
        dbc.ModalBody(id = "summary_modal_body"),
        dbc.ModalFooter(dbc.Container([download_summary, close_summary]))
    ],
    id="summary_modal",
    scrollable=True,
    is_open=False,
    size = "xl",
)

##################################
# App Layout
##################################

store_tokens = dcc.Store(id = "store_tokens")
store_suggestions = dcc.Store(id = "store_suggestions")
store_highlights = dcc.Store(id = "store_highlights")
store_summary = dcc.Store(id = "store_summary")
store_document= dcc.Store(id = "store_document")

store_scroll= html.P(0, id = "store_scroll", hidden = True)
download = dcc.Download(id = "download")

app.layout = dbc.Container(
    [
        html.H1("Query-focused Extractive Summarization"),
        html.Hr(),
        dbc.Row([controls, summary_display]),
        document_modal,
        summary_modal,
        store_highlights,
        store_suggestions,
        store_summary,
        store_document,
        store_scroll,
        store_tokens,
        download,
        visdcc.Run_js(id = 'javascript')
    ],
)

###################################
# Callbacks
###################################

@app.callback(
    Output("document_modal", "is_open"),
    [
        Input("view_document", "n_clicks"),
        Input("close_document", "n_clicks"),
        Input("store_scroll", "children")
    ],
    [State("document_modal", "is_open")],
)
def toggle_modal_document(open_view, close_view, scroll, is_open):
    if open_view or close_view or scroll:
        return not is_open
    else:
        return is_open


@app.callback(
    Output("summary_modal", "is_open"),
    [
        Input("view_summary", "n_clicks"),
        Input("close_summary", "n_clicks"),
    ],
    [State("summary_modal", "is_open")],
)
def toggle_modal_summary(open_view, close_view, is_open):
    if open_view or close_view:
        return not is_open
    else:
        return is_open


@app.callback(
    Output('javascript', 'run'),
    Input("document_modal", "is_open"),
    State("store_scroll", "children"),
    State("store_tokens", "data")
)
def scroll_document(is_open, scroll_position, tokens):
    tokens = json.loads(tokens)
    print("scroll_document")
    print("scroll_position", scroll_position)
    print("scroll_tokens", len(tokens))
    if scroll_position is not None and len(tokens) > 0:
        position = int(scroll_position) / len(tokens)
        print("position", position)
        output = f"""var obj = document.getElementById('document_modal_body');
        var line = {position} * obj.scrollHeight;
        obj.scrollTop = line"""
    else:
        output = ""
    return output 


@app.callback(
    Output("store_scroll", "children"),
    Input({"type":"highlight", "index":ALL}, "n_clicks"),
    State("store_suggestions", "data")
)
def click_highlight(*args):
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]
    value = ctx.triggered[0]["value"]
    if value is None:
        start = 0
    else:
        index_type, attribute = prop_id.split(".")
        index_type = json.loads(index_type)
        index = index_type["index"]
        type = index_type["type"]
        highlights = json.loads(args[-1])
        start = highlights[index]["start"]
    return start 


@app.callback(
    Output("store_highlights", "data"),
    Output("reviews", "children"),
    Input({"type":"accept", "index":ALL}, "n_clicks"),
    Input({"type":"reject", "index":ALL}, "n_clicks"),
    Input("store_suggestions", "data"),
    State("store_highlights", "data")
)
def review_highlight(*args):
    suggestions = json.loads(args[-2])
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]
    value = ctx.triggered[0]["value"]
    highlights = suggestions 
    reviews = 0
    if prop_id != "store_suggestions.data":
        index_type, attribute = prop_id.split(".")
        index_type = json.loads(index_type)
        index = index_type["index"]
        type = index_type["type"]
        if value is not None:
            highlights = json.loads(args[-1])
            if type == "accept":
                highlights[index]["accepted"] = True 
            elif type == "reject":
                highlights[index]["accepted"] = False 
            else:
                highlights[index]["accepted"] = None 
            reviews = len([h for h in highlights if h["accepted"] is not None])
    return json.dumps(highlights), reviews

@app.callback(
    Output({"type":"highlight_card", "index":ALL}, "style"),
    Input("store_highlights", "data"),
)
def update_highlights_color(data):
    if data is not None:
        status = json.loads(data)
    else:
        status = []
    output = []
    for h in status:
        if h["accepted"] is None:
            output.append({"background-color":"white"})
        elif h["accepted"]:
            output.append({"background-color":"lightgreen"})
        else:
            output.append({"background-color":"lightpink"})
    return output 


@app.callback(
    Output("summary_modal_body", "children"),
    Output("store_summary", "data"),
    Input("view_summary", "n_clicks"),
    State("store_highlights", "data"),
    State("store_tokens", "data")
)
def retrieve_summary(is_open, highlights, tokens):
    if highlights and tokens: 
        highlights = json.loads(highlights)
        tokens = json.loads(tokens)
        lines = ["".join(tokens[h["start"]:h["end"]]) for h in highlights if h["accepted"]]
        text = ("\n" + "-" * 30 + "\n").join(lines)
        body = lib.join(text.split("\n"))
    else:
        text = ""
        body = []
    return body, json.dumps(text) 


@app.callback(
    Output("download", "data"),
    Input("download_summary", "n_clicks"),
    State("store_summary", "data"),
    prevent_initial_call = True
)
def download(n_clicks, summary):
    if summary:
        output = json.loads(summary)
    else:
        output = ""
    return dict(content = output, filename = "summary.txt") 


@app.callback(
    Output("store_document", "data"),
    Output("store_tokens", "data"),
    Input("upload", "contents"),
)
def upload_document(contents):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        file = BytesIO(decoded)
        pdf = pt.PDF(file, raw = True)
        document = "".join(pdf)
        tokens = re.findall(r"[a-zA-Z]+|[^a-zA-Z]", document)
    else:
        document = None
        tokens = []
    return json.dumps(document), json.dumps(tokens)


@app.callback(
   Output("store_suggestions", "data"),
   Input("submit", "n_clicks"),
   State("query", "value"),
   State("store_tokens", "data"),
)
def compute_highlights(submit, query, tokens):
    if query and tokens:
        highlights = [{"start": 110, "end":120, "accepted":None}, 
                      {"start": 1530, "end":1640, "accepted":None},
                      {"start": 2530, "end":2640, "accepted":None},
                      {"start": 3530, "end":3640, "accepted":None},
                      {"start": 4530, "end":4640, "accepted":None},
                      {"start": 8530, "end":8640, "accepted":None}]
    else:
        highlights = []
    return json.dumps(highlights)


@app.callback(
    Output("summary_body", "children"),
    Input("store_suggestions", "data"),
    State("store_tokens", "data"),
)
def update_summary_display(suggestions, tokens):
    if suggestions and tokens:
        suggestions = json.loads(suggestions)
        tokens = json.loads(tokens)
        summary_body = []
        for i, hl in enumerate(suggestions):
            content = dbc.Button("".join(tokens[hl["start"]:hl["end"]]), 
                                 id = dict(type = "highlight", index = i), 
                                 color = "link")
            reject = dbc.Button(f"✕", 
                                id = dict(type = "reject", index = i), 
                                style = {"float":"right", "background":"firebrick"})
            accept = dbc.Button(f"✔", 
                                id = dict(type = "accept", index = i), 
                                style = {"float":"right", "background":"seagreen"})
            row = dbc.Card(dbc.Container([content, reject, accept], fluid = True), 
                           id = dict(type = "highlight_card", index = i))
            summary_body.append(row)
    else:
        summary_body = []
    return summary_body


@app.callback(
    Output("document_modal_body", "children"),
    Input("store_tokens", "data"),
    Input("store_suggestions", "data"),
)
def update_document_modal(tokens, suggestions):
    if tokens and suggestions:
        tokens = json.loads(tokens)
        suggestions = json.loads(suggestions)
        output = lib.render_document(tokens, suggestions)
    else:
        output = None
    return output

if __name__ == '__main__':
    app.run_server(debug=True)
