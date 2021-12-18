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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server = server)

###################################
# Controls
###################################

query_test = """In this paper, we propose a novel neural network model called RNN Encoder-Decoder that consists of two recurrent neural networks (RNN). One RNN encodes a sequence of symbols into a fixed-length vector representation, and the other decodes the representation into another sequence of symbols. The encoder and decoder of the proposed model are jointly trained to maximize the conditional probability of a target sequence given a source sequence. The performance of a statistical machine translation system is empirically found to improve by using the conditional probabilities of phrase pairs computed by the RNN Encoder-Decoder as an additional feature in the existing log-linear model. Qualitatively, we show that the proposed model learns a semantically and syntactically meaningful representation of linguistic phrases."""
query = dbc.Card([html.H2("Query"), 
                  dbc.Card(dcc.Textarea(rows = 9, id = "query", 
                                        value = query_test, 
                                        style = {"font-size":"12px"}))], 
                 body = True)
view_document = dbc.Card(dbc.Button("View document", id = "view_document"))
upload = dbc.Card(dbc.Button(dcc.Upload('Upload paper', id = "upload")))
buttons = dbc.Row([dbc.Col(upload), dbc.Col(view_document)])
threshold = dbc.Card([html.H3("Threshold")])
row1 = html.Tr([html.Td(html.Strong("Window size:")), 
                html.Td(id = "window_size_front", 
                        style = {"text-align":"right"})])
row2 = html.Tr([html.Td(html.Strong("Threshold:")),
                html.Td(id = "threshold_front", 
                        style = {"text-align":"right"})])
parameters_body = [row1, row2]
parameters = html.Table(parameters_body)
settings_adjust = dbc.Card(dbc.Button("Adjust settings", 
                                      id = "settings_adjust"))
settings_panel = dbc.Card([html.H2("Settings"), parameters], body = True)
controls = dbc.Col([query, buttons, settings_panel, settings_adjust], 
                   width = 4)

###################################
# Summary
###################################

highlighted_words_front = html.Td(0, id = "highlighted_words_front", 
                                  style = {"text-align":"right"})
highlighted_lines_front = html.Td(0, id = "highlighted_lines_front", 
                                  style = {"text-align":"right"})
row1 = html.Tr([html.Td(html.Strong("Highlighted words: ")), 
                highlighted_words_front])
row2 = html.Tr([html.Td(html.Strong("Highlights: ")), 
                highlighted_lines_front])
row3 = html.Tr([html.Td(html.Strong("Reviewed: ")), 
                html.Td(0, 
                        id = "reviews", 
                        style = {"text-align":"right"})])
counts_body = [row1, row2, row3]
counts = dbc.Card(html.Table(counts_body), body = True)
summary_title = dbc.CardGroup([dbc.Card(html.H2("Highlights"), body = True), 
                               counts])
view_summary = dbc.Button(
    "View summary",
    id="view_summary",
    n_clicks=0,
)
summary_body = dbc.Card(id = "summary_body",
                        style = {"height":"330px", 
                                 "overflow-y":"scroll", 
                                 "background-color":"lightgrey"})
summary_display = dbc.Col([summary_title, 
                           summary_body, 
                           dbc.Card(view_summary)])

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
                                style = {"height":"400px", 
                                         "overflow-y":"scroll"})]),
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
# Settings modal
##################################
histogram = dbc.Col(dcc.Graph(id = "histogram"))
lineplot = dbc.Col(dcc.Graph(id = "lineplot"))
window_size_row = [html.Td(html.Strong("Window size:")), 
                   html.Td(dcc.Input(value = 64, 
                                     style = {"text-align":"right"},
                                     id = "window_size"),)]
                           #style = {"text-align":"right"})]
threshold_row = [html.Td(html.Strong("Threshold:")), 
                 html.Td(dcc.Input(value = 0.5, 
                         style = {"text-align":"right"}, 
                         id = "threshold"), )]
highlighted_words_modal = html.Td(0, id = "highlighted_words_modal", 
                                  style = {"text-align":"right"})
highlighted_lines_modal = html.Td(0, id = "highlighted_lines_modal", 
                                  style = {"text-align":"right"})
highlighted_words_row = [html.Td(html.Strong("Highlighted words:")), 
                         highlighted_words_modal]
highlighted_lines_row = [html.Td(html.Strong("Highlights:")), 
                         highlighted_lines_modal]
parameters = dbc.Card(html.Table([html.Tr(window_size_row + \
                                          highlighted_words_row),
                                  html.Tr(threshold_row + \
                                          highlighted_lines_row)]))
settings = html.Div([parameters, dbc.Row([histogram, lineplot])])
settings_close = dbc.Button("Close", id = "settings_close")
submit = dbc.Card(dbc.Button("Compute word scores", id = "submit"))

settings_modal = dbc.Modal(
    [
        dbc.ModalHeader(html.H1("Settings")),
        dbc.ModalBody(settings),
        dbc.ModalFooter([submit, settings_close])
    ],
    id = "settings_modal",
    is_open = False,
    scrollable = True,
    size = "xl"
)

##################################
# App Layout
##################################

store_tokens = dcc.Store(id = "store_tokens")
store_word_scores = dcc.Store(id = "store_word_scores")
store_highlights = dcc.Store(id = "store_highlights")
store_summary = dcc.Store(id = "store_summary")
store_document= dcc.Store(id = "store_document")

store_scroll= html.P(0, id = "store_scroll", hidden = True)
store_highlighted_words = html.P(0, id = "store_highlighted_words", 
                                 hidden = True)
store_highlighted_lines = html.P(0, id = "store_highlighted_lines", 
                                 hidden = True)
download = dcc.Download(id = "download")

app.layout = dbc.Container(
    [
        html.H1("Query-focused Extractive Summarization"),
        html.Hr(),
        dbc.Row([controls, summary_display]),
        document_modal,
        summary_modal,
        settings_modal,
        store_tokens,
        store_word_scores,
        store_highlights,
        store_summary,
        store_document,
        store_scroll,
        store_highlighted_words,
        store_highlighted_lines,
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
    Output("settings_modal", "is_open"),
    [
        Input("settings_adjust", "n_clicks"),
        Input("settings_close", "n_clicks"),
    ],
    [State("settings_modal", "is_open")],
)
def toggle_settings_modal(open_settings, close_settings, is_open):
    if open_settings or close_settings:
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
    if scroll_position is not None and len(tokens) > 0:
        position = int(scroll_position) / len(tokens)
        output = f"""var obj = document.getElementById('document_modal_body');
        var line = {position} * obj.scrollHeight;
        obj.scrollTop = line"""
    else:
        output = ""
    return output 


@app.callback(
    Output("store_scroll", "children"),
    Input({"type":"highlight", "index":ALL}, "n_clicks"),
    State("store_highlights", "data")
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
    Input("store_word_scores", "data"),
    Input("threshold", "value"),
    State("store_highlights", "data"),
)
def update_highlights(*args):
    word_scores = args[-3]
    threshold = args[-2]
    if word_scores:
        word_scores = pd.read_json(word_scores)
        threshold = float(threshold)
        spans = lib.find_spans(word_scores.avg_score, threshold = threshold)
        highlights = [dict(start = s, end = e, accepted = None) 
                      for s, e in spans]
    else:
        highlights = []
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]
    value = ctx.triggered[0]["value"]
    reviews = 0              # 
    if prop_id not in ["store_word_scores.data", "threshold.value"]:
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
            reviews = len([h for h in highlights 
                           if h["accepted"] is not None])
    return json.dumps(highlights), reviews

@app.callback(
    Output({"type":"highlight_card", "index":ALL}, "style"),
    Input("summary_body", "children"),
    State("store_highlights", "data"),
)
def update_highlights_color(children, data):
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
        lines = ["".join(tokens[h["start"]:h["end"]]) 
                 for h in highlights if h["accepted"]]
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
    Output("summary_body", "children"),
    Input("store_highlights", "data"),
    State("store_tokens", "data"),
    State("store_word_scores", "data"),
    State("window_size", "value"),
    State("query", "value"),
)
def update_summary_display_dl(highlights, tokens, word_scores, window_size, 
                              query):
    if highlights and tokens and word_scores and window_size and query:
        highlights = json.loads(highlights)
        tokens = json.loads(tokens)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        highlight_text = ["".join(tokens[hl["start"]:hl["end"]]) 
                          for hl in highlights]
        if len(highlight_text) > 0:
            highlight_embeddings = model.encode(highlight_text)
            similarities = cosine_similarity(query_embedding, 
                                             highlight_embeddings)\
                .flatten().tolist()
        else:
            similarities = []
        summary_body = []
        for i, hl in enumerate(highlights):
            text = "".join(tokens[hl["start"]:hl["end"]])
            score = html.Td(round(similarities[i], 3))
            content = html.Td(dbc.Button(text, 
                                 id = dict(type = "highlight", index = i), 
                                 color = "link"))
            accept = dbc.Button(f"✔", 
                                id = dict(type = "accept", index = i), 
                                style = {"background":"seagreen"})
            reject = dbc.Button(f"✕", 
                                id = dict(type = "reject", index = i), 
                                style = {"background":"firebrick"})

            row = html.Tr([score, content, html.Td([accept, reject])], 
                           id = dict(type = "highlight_card", index = i))
            summary_body.append(row)
    else:
        summary_body = []
    return dbc.Table(summary_body, bordered = True)

@app.callback(
    Output("document_modal_body", "children"),
    Input("store_tokens", "data"),
    Input("store_highlights", "data"),
)
def update_document_modal(tokens, highlights):
    tokens = json.loads(tokens)
    highlights = json.loads(highlights)
    output = lib.render_document(tokens, highlights)
    return output


@app.callback(
    Output("store_word_scores", "data"),
    Input("submit", "n_clicks"),
    State("query", "value"),
    State("store_tokens", "data"),
    State("window_size", "value"),
)
def compute_word_scores(submit, query, tokens, window_size):
    window_size = int(window_size)
    output = ""
    if query and tokens:
        tokens = json.loads(tokens)
        word_scores = lib.compute_scores_sentence_dl(tokens, query, 
                                                     window_size = window_size)
        output = word_scores.to_json()
        return output

@app.callback(
    Output("histogram", "figure"),
    Output("lineplot", "figure"),
    Input("threshold", "value"),
    Input("store_word_scores", "data"),
)
def update_plots(threshold, word_scores):
    histogram = px.histogram()
    lineplot = px.scatter()
    words = 0
    lines = 0
    threshold = float(threshold) 
    if word_scores:
        word_scores = pd.read_json(word_scores)
        histogram = px.histogram(word_scores, "avg_score")
        lineplot = px.line(word_scores, 
                              y = "avg_score")
        lineplot.update_traces(marker={'size': 1})
        histogram.add_vline(x = threshold, annotation_text = "Threshold")
        histogram.update_layout(xaxis_title = "Word score", 
                                yaxis_title = "Frequency", 
                                title = "Distribution of the word scores",
                                xaxis_range = (0, 1))
        lineplot.update_layout(xaxis_title = "Position in document", 
                               yaxis_title = "Word score", 
                               title = "Evolution of the word score in the paper",
                               yaxis_range = (0, 1))
        lineplot.add_hline(y = threshold, annotation_text = "Threshold")
    return histogram, lineplot

@app.callback(
    Output("highlighted_words_front", "children"),
    Output("highlighted_lines_front", "children"),
    Output("highlighted_words_modal", "children"),
    Output("highlighted_lines_modal", "children"),
    Input("store_highlights", "data"),
)
def update_highlighted(highlights):
    highlights = json.loads(highlights)
    words = sum([h["end"] - h["start"] for h in highlights])
    lines = len(highlights)
    return f"{words:,}", f"{lines:,}", f"{words:,}", f"{lines:,}" 


@app.callback(
    Output("window_size_front", "children"),
    Output("threshold_front", "children"),
    Input("window_size", "value"),
    Input("threshold", "value"),
)
def display_parameters(window_size, threshold):
    return window_size, threshold


if __name__ == '__main__':
    app.run_server(debug=True)
