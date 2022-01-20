import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
import pdftotext as pt
from flask import Flask
from io import BytesIO
import visdcc
import re
import json
import base64
import lib
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import PunktSentenceTokenizer
import numpy as np
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px

from dash.long_callback import DiskcacheLongCallbackManager
import diskcache

server = Flask(__name__)
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server = server)

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], server = server, long_callback_manager=long_callback_manager)

###################################
# Controls
###################################

query_test = """In this paper, we propose a novel neural network model called RNN Encoder-Decoder that consists of two recurrent neural networks (RNN). One RNN encodes a sequence of symbols into a fixed-length vector representation, and the other decodes the representation into another sequence of symbols. The encoder and decoder of the proposed model are jointly trained to maximize the conditional probability of a target sequence given a source sequence. The performance of a statistical machine translation system is empirically found to improve by using the conditional probabilities of phrase pairs computed by the RNN Encoder-Decoder as an additional feature in the existing log-linear model. Qualitatively, we show that the proposed model learns a semantically and syntactically meaningful representation of linguistic phrases."""
query = dbc.Card([html.H2("Query"), 
                  dcc.Textarea(id = "query", value = query_test, rows = 8)], 
                 body = True)
view_document = dbc.Card(dbc.Button("View document", id = "view_document"))
upload = dbc.Card(dbc.Button(dcc.Upload('Upload paper', id = "upload")))
settings_adjust = dbc.Card(dbc.Button("Adjust settings", id = "settings_adjust"))
view_summary = dbc.Card(dbc.Button("View summary", id="view_summary"))
buttons = dbc.Card([dbc.CardGroup([upload, view_document]), dbc.CardGroup([settings_adjust, view_summary])])
highlighted_words_front = html.Td(0, id = "highlighted_words_front", 
                                  style = {"text-align":"right"})
highlighted_lines_front = html.Td(0, id = "highlighted_lines_front", 
                                  style = {"text-align":"right"})


# summary_title = dbc.CardGroup([dbc.Card(html.H2("Highlights"), body = True), 
#                                counts])

threshold = dbc.Card([html.H3("Threshold")])
row1 = html.Tr([html.Td(html.Strong("Window size:")), 
                html.Td(id = "window_size_front", 
                        style = {"text-align":"right"})])
row2 = html.Tr([html.Td(html.Strong("Threshold:")),
                html.Td(id = "threshold_front", 
                        style = {"text-align":"right"})])
row3 = html.Tr([html.Td(html.Strong("Highlighted words: ")), 
                highlighted_words_front])
row4 = html.Tr([html.Td(html.Strong("Highlights: ")), 
                highlighted_lines_front])
row5 = html.Tr([html.Td(html.Strong("Reviewed: ")), 
                html.Td(0, 
                        id = "reviews", 
                        style = {"text-align":"right"})])
parameters_body = [row1, row2, row3, row4, row5]
parameters = html.Table(parameters_body)


settings_panel = dbc.Card([html.H2("Settings"), parameters], body = True)
controls = dbc.Row([dbc.Col(query, width = 8), dbc.Col([settings_panel, buttons])])

###################################
# Summary
###################################


summary_body = dbc.Card(id = "summary_body")
summary_display = summary_body

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
        dbc.ModalBody(id = "history_body"),
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

store_sentences = dcc.Store(id = "store_sentences")
store_sentence_embeddings = dcc.Store(id = "store_sentence_embeddings")
store_query_embedding= dcc.Store(id = "store_query_embedding")
history = pd.DataFrame(columns = ["sentence", "text", "relevance"]).to_json()
store_history = dcc.Store(data = history, 
                          id = "store_history")
store_recommendations = dcc.Store(id = "store_recommendations")


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
        controls, 
        summary_display,
        document_modal,
        summary_modal,
        settings_modal,
        store_sentences,
        store_sentence_embeddings,
        store_query_embedding,
        store_history,
        store_recommendations,
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
    State("store_sentences", "data"),
    prevent_initial_call = True
)
def scroll_document(is_open, store_scroll, sentences):
    if is_open and sentences and store_scroll:
        sentences = json.loads(sentences)
        position = (int(store_scroll) + 5) / len(sentences)
        javascript = f"""var obj = document.getElementById('document_modal_body');
        var line = {position} * obj.scrollHeight;
        obj.scrollTop = line"""
    else:
        javascript = ""
    return javascript 


@app.callback(
    Output("store_scroll", "children"),
    Input({"type":"recommendation", "index":ALL}, "n_clicks"),
    Input({"type":"history", "index":ALL}, "n_clicks"),
    State("store_scroll", "children"),
    State("store_recommendations", "data"),
    State("store_history", "data"),
    prevent_initial_call = True
)
def update_scroll(*args):
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]
    value = ctx.triggered[0]["value"]
    store_scroll = args[-3]
    position = store_scroll
    if value:
        recommendations = pd.read_json(args[-2])
        history = pd.read_json(args[-1])
        index_type, attribute = prop_id.split(".")
        index_type = json.loads(index_type)
        index = index_type["index"]
        type = index_type["type"]
        if type == "recommendation":
            position = recommendations.iloc[index, 0]
        else:
            position = history.iloc[index, 0]
        return position


@app.callback(
    Output("document_modal", "is_open"),
    Input("view_document", "n_clicks"),
    Input("close_document", "n_clicks"),
    Input("store_scroll", "children"),
    State("document_modal", "is_open"),
    prevent_initial_call = True
)
def toggle_modal_document(view_document, close_document, store_scroll, is_open):
    ctx = dash.callback_context
    value = ctx.triggered[0]["value"]
    if value:
        return not is_open
    

@app.callback(
    Output("download", "data"),
    Input("download_summary", "n_clicks"),
    State("store_history", "data"),
    prevent_initial_call = True
)
def download(n_clicks, history):
    if history:
        output = "\n\n".join(pd.read_json(history).query("relevance == True").text)
    else:
        output = ""
    return dict(content = output, filename = "summary.txt") 


@app.long_callback(
    Output("store_sentences", "data"),
    Output("store_sentence_embeddings", "data"),
    Output("store_query_embedding", "data"),
    Input("upload", "contents"),
    State("query", "value"),
    running = [(Output("upload", "disabled"), True, False)]
)
def upload_document(contents, query):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        file = BytesIO(decoded)
        pdf = pt.PDF(file, raw = True)
        document = "".join(pdf).replace("-\n", "").replace("\n", " ")
        tokenizer = PunktSentenceTokenizer(document)
        sentences = tokenizer.tokenize(document)
        model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder = "sbert_cache")
        sentence_embeddings = model.encode(sentences, normalize_embeddings = True).tolist()
        query_embedding = model.encode([query], normalize_embeddings = True).tolist()
    else:
        sentences = []
        sentence_embeddings = []
        query_embedding = []
    return json.dumps(sentences), json.dumps(sentence_embeddings), json.dumps(query_embedding) 


@app.callback(
    Output("summary_body", "children"),
    Input("store_recommendations", "data"),
    prevent_initial_call = True
)
def update_recommendations_body(recommendations):
    if recommendations:
        recommendations = pd.read_json(recommendations).head().to_dict("records")
        summary_body = []
        header = html.Tr([html.Th("Estimated Relevance", style = {"text-align":"center"}),
                          html.Th("Sentence", style = {"text-align":"center"}),
                          html.Th("Text", style = {"text-align":"center"}),
                          html.Th("Relevant?", colSpan = 2, style = {"text-align":"center"})])
        summary_body.append(header)
        for i, r in enumerate(recommendations):
            score = html.Td(round(r["score"], 3))
            sentence = html.Td(r["sentence"])
            content = html.Td(dbc.Button(r["text"], 
                                  id = dict(type = "recommendation", index = i), 
                                  color = "link"))
            accept = dbc.Button("✔", 
                                id = dict(type = "recommendation_accept", 
                                          index = i), 
                                style = {"background":"seagreen"})
            reject = dbc.Button("✕", 
                                id = dict(type = "recommendation_reject", 
                                          index = i), 
                                style = {"background":"firebrick"})

            row = html.Tr([score, sentence, content, html.Td(accept), html.Td(reject)], 
                          id = dict(type = "recommendation_card", index = i), 
                          style = {"background":"white"})
            summary_body.append(row)
    else:
        summary_body = []
    return dbc.Table(summary_body, bordered = True)


@app.callback(
    Output("history_body", "children"),
    Input("store_history", "data"),
    prevent_initial_call = True
)
def update_history_body(history):
    history = pd.read_json(history).to_dict("records")
    history_body = []
    for i, r in enumerate(history):
        sentence = html.Td(r["sentence"])
        content = html.Td(dbc.Button(r["text"], 
                              id = dict(type = "history", index = i), 
                              color = "link"))
        accept = dbc.Button("✔", 
                            id = dict(type = "history_accept", index = i), 
                            style = {"background":"seagreen"})
        reject = dbc.Button("✕", 
                            id = dict(type = "history_reject", index = i), 
                            style = {"background":"firebrick"})
        card_style = {"background":"lightgreen"} if r["relevance"] else {"background":"lightpink"} 
        row = html.Tr([sentence, content, html.Td(accept), html.Td(reject)], 
                      id = dict(type = "history_card", index = i), 
                      style = card_style)
        history_body.append(row)
    return  dbc.Table(history_body, bordered = True)


@app.callback(
    Output("store_history", "data"),
    Input({"type":"recommendation_accept", "index":ALL}, "n_clicks"),
    Input({"type":"recommendation_reject", "index":ALL}, "n_clicks"),
    Input({"type":"history_accept", "index":ALL}, "n_clicks"),
    Input({"type":"history_reject", "index":ALL}, "n_clicks"),
    State("store_history", "data"),
    State("store_recommendations", "data"),
    prevent_initial_call = True
)
def update_history(*args):
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]
    value = ctx.triggered[0]["value"]
    index_type, attribute = prop_id.split(".")
    index_type = json.loads(index_type)
    index = index_type["index"]
    type = index_type["type"]
    subtype, accept = type.split("_")
    history = pd.read_json(args[-2])
    recommendations = args[-1]
    if value and recommendations:
        recommendations = pd.read_json(recommendations).to_dict("records")
        if subtype == "recommendation":
            r = recommendations[index]
            new = dict(sentence = r["sentence"], 
                        text = r["text"], 
                        relevance = True if accept == "accept" else False)
            history = history.append(pd.Series(new), ignore_index = True)
        else:
            history.iloc[index, -1] = True if accept == "accept" else False
    return history.sort_values(["relevance", "sentence"], ascending = [0, 1]).to_json()


@app.callback(
    Output("document_modal_body", "children"),
    Input("store_sentences", "data"),
    Input("store_history", "data"),
    Input("store_scroll", "children"),
    prevent_initial_call = True
)
def update_document_modal(sentences, history, scroll):
    sentences = json.loads(sentences)
    history = pd.read_json(history)
    output = lib.render_document(sentences, history, scroll)
    return output


@app.callback(
    Output("store_recommendations", "data"),
    Input("store_sentences", "data"),
    Input("store_history", "data"),
    State("store_sentence_embeddings", "data"),
    State("store_query_embedding", "data"),
    prevent_initial_call = True,
)
def update_recommendations(sentences, history, sentence_embeddings, query_embedding):
    sentences = json.loads(sentences)
    sentence_embeddings = np.array(json.loads(sentence_embeddings))
    query_embedding = np.array(json.loads(query_embedding))
    history = pd.read_json(history)
    if len(sentences) > 0:
        recommendations = lib.compute_scores_dl(sentences, history, sentence_embeddings, query_embedding)
        recommendations = recommendations.sort_values("score", 
                                                      ascending = False)
    else:
        recommendations = pd.DataFrame()
    return recommendations.to_json()


@app.callback(
    Output("histogram", "figure"),
    Output("lineplot", "figure"),
    Input("threshold", "value"),
    Input("store_recommendations", "data"),
    Input("store_history", "data"),
)
def update_plots(threshold, recommendations, history):
    histogram = px.histogram()
    lineplot = px.scatter()
    threshold = float(threshold) 
    if recommendations and history:
        recommendations = pd.read_json(recommendations)
        history = pd.read_json(history)\
            .assign(score = lambda df: df.relevance.map(float))
        data = pd.concat((recommendations, history), axis = 0)\
            .reset_index(drop = True)\
            .sort_values("sentence")
        histogram = px.histogram(data, "score", nbins = 50)
        lineplot = px.line(data, x = "sentence", y = "score")
        lineplot.update_traces(marker={'size': 1})
        histogram.add_vline(x = threshold, annotation_text = "Threshold")
        histogram.update_layout(xaxis_title = "Sentence score", 
                                yaxis_title = "Frequency", 
                                title = "Distribution of the sentence scores",
                                xaxis_range = (-0.1, 1.1))
        lineplot.update_layout(xaxis_title = "Position in document", 
                                yaxis_title = "Sentence score", 
                                title = "Evolution of the sentence score in the paper",
                                yaxis_range = (-0.1, 1.1))
        lineplot.add_hline(y = threshold, annotation_text = "Threshold")
    return histogram, lineplot


@app.callback(
    Output("highlighted_words_front", "children"),
    Output("highlighted_lines_front", "children"),
    Output("highlighted_words_modal", "children"),
    Output("highlighted_lines_modal", "children"),
    Output("reviews", "children"),
    Input("store_history", "data"),
)
def update_highlighted(history):
    history = pd.read_json(history)
    relevant = history.query("relevance == True")
    words = len(re.findall(r"\w+", 
                           " ".join(relevant.text)))
    lines = relevant.shape[0]
    reviews = history.shape[0]
    return f"{words:,}", f"{lines:,}", f"{words:,}", f"{lines:,}", f"{reviews:,}" 


@app.callback(
    Output("window_size_front", "children"),
    Output("threshold_front", "children"),
    Input("window_size", "value"),
    Input("threshold", "value"),
)
def display_parameters(window_size, threshold):
    return window_size, threshold


if __name__ == '__main__':
    app.run_server(debug=True, host = "0.0.0.0")
