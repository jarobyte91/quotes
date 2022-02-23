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
from sentence_transformers import SentenceTransformer
from nltk.tokenize import PunktSentenceTokenizer
import numpy as np
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.preprocessing import normalize
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache

server = Flask(__name__)
cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
app = dash.Dash(
    __name__, 
    external_stylesheets = [dbc.themes.BOOTSTRAP], 
    server = server, 
    long_callback_manager = long_callback_manager
)
app.title = "QuOTeS"

###################################
# Upload tab
###################################

store_papers = dcc.Store(id = "store_papers")
add_paper = html.Table(
    html.Tr(
        [
            html.Td(
                dbc.Button(
                    children = dcc.Upload(id = "add_paper", children = "Upload file")
                )
            ),
            html.Td("    "),
            html.Td("file:"),
            html.Td(id = "filename"), 
            html.Td("    "),
            html.Td(dbc.Button(id = "process", children = "Add Paper"))
        ]
    )
)
tab_upload = dbc.Tab(
    label = "Upload", 
    id = "tab_upload", 
    label_style = {"font-size":"1.5em"},
    children = dbc.Container(
        [
            store_papers,
            html.P(),
            add_paper,
            html.P(),
            html.H3("Uploaded Papers"),
            html.Div(id = "paper_list_show")
        ], 
        fluid = True
    )
)

###################################
# Settings tab
###################################

tab_settings = dbc.Tab(
    label = "Settings", 
    label_style = {"font-size":"1.5em"},
    children = dbc.Container(
        dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem("Deep Learning", header = True),
                dbc.DropdownMenuItem("all-MiniLM-L6-v2"),
                dbc.DropdownMenuItem(divider = True),
                dbc.DropdownMenuItem("Classical", header = True),
                dbc.DropdownMenuItem("Character Trigrams"),
                dbc.DropdownMenuItem(divider = True),
            ],
            label = "Model"
        ),
        fluid = True,
    )
)

###################################
# Reviews tab
###################################

tab_reviews = dbc.Tab(
    label = "Reviews", 
    label_style = {"font-size":"1.5em"},
    children = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.H3("Query")),
                    dbc.Col(),
                    dbc.Col(
                        dbc.Card(dbc.Button("Download .csv", id = "download_csv")),
                        width = 2
                    )
                ]
            ),
            html.P(id = "reviews_query"),
            html.H3("Reviewed Sentences"),
            html.Div(id = "history_body")
        ],
        fluid = True
    )
)

###################################
# Summary tab
###################################

summary_body = dbc.Container(id = "summary_body", fluid = True)
summary_display = summary_body
highlighted_words_front = html.Td(
    0, 
    id = "highlighted_words_front", 
    style = {"text-align":"right"}
)
highlighted_lines_front = html.Td(
    0, 
    id = "highlighted_lines_front", 
    style = {"text-align":"right"}
)
words = html.Tr(
    [
        html.Td(html.Strong("Highlighted words: ")), 
        highlighted_words_front
    ]
)
highlights = html.Tr(
    [
        html.Td(html.Strong("Highlights: ")), 
        highlighted_lines_front
    ]
)
reviews = html.Tr(
    [
        html.Td(html.Strong("Reviewed: ")), 
        html.Td(
            0, 
            id = "reviews", 
            style = {"text-align":"right"}
        )
    ]
)
counts = dbc.Table([words, highlights, reviews])
tab_summary = dbc.Tab(
    label = "Summary", 
    label_style = {"font-size":"1.5em"},
    children = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col([html.H3("Highlight Counts"), counts], width = 3),
                dbc.Col(),
                dbc.Col(
                    dbc.Card(dbc.Button("Download .txt", id = "download_txt")), 
                    width = 2
                )
            ]
        ),
        html.H3("Query"),
        html.P(id = "summary_query"),
        html.H3("Accepted Sentences"),
        html.Ul(id = "accepted_sentences")
    ],
    fluid = True
    ), 
)

###################################
# Highlights tab
###################################

query = dbc.Container(
    children = [
        dbc.Row(
            [
                dbc.Col(html.H3("Query")),
                dbc.Col(dbc.Card(dbc.Button("Submit", id = "submit")), width = 2)
            ]
        ),
        dbc.Textarea(
            id = "query", 
            rows = 5
        ),
        html.H3("Relevant Sentences")
    ], 
    fluid = True
)
tab_highlights = dbc.Tab(
    label = "Highlights", 
    label_style = {"font-size":"1.5em"}, 
    children =[
        query,
        summary_display,
    ],
)

###################################
# Documents tab
###################################

tab_documents = dbc.Tab(
    label = "Documents",
    label_style = {"font-size":"1.5em"},
    children = dbc.Container(
        [
            dcc.Graph(id = "barplot"),
            dcc.Graph(id = "histogram"),
            dcc.Graph(id = "boxplot"),
        ],
        fluid = True
    )

)

###################################
# Stores
###################################

store_sentences = dcc.Store(id = "store_sentences")
store_sentence_embeddings = dcc.Store(id = "store_sentence_embeddings")
store_query_embedding= dcc.Store(id = "store_query_embedding")
history = pd.DataFrame(
    columns = ["paper", "sentence", "text", "relevance"]
).to_json()
store_history = dcc.Store(
    data = history, 
    id = "store_history"
)
store_recommendations = dcc.Store(id = "store_recommendations")
store_scroll= html.P(0, id = "store_scroll", hidden = True)
store_highlighted_words = html.P(
    0, 
    id = "store_highlighted_words", 
    hidden = True
)
store_highlighted_lines = html.P(
    0, 
    id = "store_highlighted_lines", 
    hidden = True
)
download = dcc.Download(id = "download")
store = html.Div(
    [
        store_sentences,
        store_sentence_embeddings,
        store_query_embedding,
        store_history,
        store_recommendations,
        store_scroll,
        store_highlighted_words,
        store_highlighted_lines,
        visdcc.Run_js(id = 'javascript'),
        download
    ]
)

###################################
# Main Container 
###################################

app.layout = dbc.Container(
    fluid = True, 
    children = [
        html.H1("QuOTeS - Query-Oriented Technical Summarization"),
        store,
        dbc.Tabs(
            [
                tab_upload, 
                tab_highlights, 
                tab_reviews, 
                tab_documents, 
                tab_summary
            ]
        ),
    ]
)
            
###################################
# Callbacks
###################################

@app.callback(
        Output("paper_list_show", "children"), 
        Input("store_papers", "data"),
        State("store_sentences", "data"),
        prevent_initial_call = True
    )
def show_papers(papers, sentences):
    papers = pd.read_json(papers)
    sentences = pd.read_json(sentences)\
    .groupby("paper")["sentence"]\
    .size()\
    .reset_index()
    df = papers.merge(sentences)
    output = []
    if len(df) > 0:
        rows = []
        header = html.Tr(
            [
                html.Th("File"), 
                html.Th("Text"), 
                html.Th("Characters"), 
                html.Th("Sentences")
            ]
        )
        rows.append(header)
        for filename, paper, text, sentences in df.values:
            r = html.Tr(
                [
                    html.Td(filename),
                    html.Td(text[:200] + "..."), 
                    html.Td(f"{len(text):,}"), 
                    html.Td(sentences)
                ]
            )
            rows.append(r)
        output = dbc.Table(rows)
    return output


@app.callback(
    Output("filename", "children"),
    Input("add_paper", "contents"),
    State("add_paper", "filename")
) 
def show_filename(contents, filename):
    return filename


@app.callback(
    Output("store_papers", "data"),
    Output("store_sentences", "data"),
    Input("process", "n_clicks"),
    State("add_paper", "contents"),
    State("store_papers", "data"),
    State("store_sentences", "data"),
    State("add_paper", "filename"),
    prevent_initial_call = True
    )
def process_paper(clicks, contents, store_papers, store_sentences, filename):
    if store_papers and store_sentences:
        store_papers = pd.read_json(store_papers)
        store_sentences = pd.read_json(store_sentences)
    else:
        store_papers = pd.DataFrame(columns = ["filename","paper", "text"])
        store_sentences = pd.DataFrame(columns = ["paper", "sentence", "text"])
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        file = BytesIO(decoded)
        pdf = pt.PDF(file, raw = True)
        document = "".join(pdf).replace("-\n", "").replace("\n", " ")
        tokenizer = PunktSentenceTokenizer(document)
        sentences = tokenizer.tokenize(document)
        new_paper = pd.DataFrame(
            [
                {
                    "filename":filename,
                    "paper":len(store_papers), 
                    "text":document
                }
            ]
        )
        store_papers = pd.concat(
            (store_papers, new_paper), 
            ignore_index = True, 
            axis = 0
        )
        sentences_df = pd.DataFrame(
            [(len(store_papers) - 1, i, s) for i, s in enumerate(sentences)], 
            columns = ["paper", "sentence", "text"]
        )
        store_sentences = pd.concat(
            (store_sentences, sentences_df), 
            ignore_index = True
        )
    return store_papers.to_json(), store_sentences.to_json()


@app.callback(
    Output("download", "data"),
    Input("download_txt", "n_clicks"),
    Input("download_csv", "n_clicks"),
    State("store_history", "data"),
    prevent_initial_call = True
)
def download(clicks_txt, clicks_csv, history):
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    content = ""
    filename = "empty"
    if history:
        history = pd.read_json(history)
        if prop_id == "download_txt":
            content = "\n\n".join(history.query("relevance == True").text)
            filename = "summary.txt"
        elif prop_id == "download_csv":
            content = history.to_csv()
            filename = "reviews.csv"
    return dict(content = content, filename = filename) 


@app.callback(
    Output("summary_body", "children"),
    Input("store_recommendations", "data"),
    prevent_initial_call = True
)
def update_recommendations_body(recommendations):
    if recommendations:
        recommendations = pd.read_json(recommendations).head().to_dict("records")
        summary_body = []
        header = html.Tr(
            [
                html.Th(
                    ["Estimated", html.Br(), "Relevance"], 
                    style = {"text-align":"center"}
                ),
                html.Th("Paper", style = {"text-align":"center"}),
                html.Th("Sentence", style = {"text-align":"center"}),
                html.Th("Text", style = {"text-align":"center"}),
                html.Th("Relevant?", colSpan = 2, style = {"text-align":"center"})
            ]
        )
        summary_body.append(header)
        for i, r in enumerate(recommendations):
            score = html.Td(round(r["score"], 3))
            paper = html.Td(r["paper"])
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

            row = html.Tr(
                [score, paper, sentence, content, html.Td(accept), html.Td(reject)], 
                id = dict(type = "recommendation_card", index = i), 
                style = {"background":"white"}
            )
            summary_body.append(row)
    else:
        summary_body = []
    return dbc.Table(summary_body)


@app.callback(
    Output("history_body", "children"),
    Input("store_history", "data"),
    prevent_initial_call = True
)
def update_history_body(history):
    history = pd.read_json(history).to_dict("records")
    header = html.Thead(
        [
            html.Th("Paper"), 
            html.Th("Sentence"), 
            html.Th("Text"), 
            html.Th("Relevant?", colSpan = 2)
        ]
    )
    history_body = [header]
    for i, r in enumerate(history):
        paper = html.Td(r["paper"])
        sentence = html.Td(r["sentence"])
        content = html.Td(
            dbc.Button(r["text"], 
            id = dict(
                type = "history", 
                index = i
            ), 
            color = "link")
        )
        accept = dbc.Button(
            "✔", 
            id = dict(type = "history_accept", index = i), 
            style = {"background":"seagreen"}
        )
        reject = dbc.Button(
            "✕", 
            id = dict(type = "history_reject", index = i), 
            style = {"background":"firebrick"}
        )
        card_style = {"background":"lightgreen"} if r["relevance"] else {"background":"lightpink"} 
        row = html.Tr([paper, sentence, content, html.Td(accept), html.Td(reject)], 
                      id = dict(type = "history_card", index = i), 
                      style = card_style)
        history_body.append(row)
    return  dbc.Table(history_body)


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
        recommendations = pd.read_json(recommendations)
        if subtype == "recommendation":
            r = recommendations.to_dict("records")[index]
            new = dict(
                paper = r["paper"],
                sentence = r["sentence"], 
                text = r["text"], 
                relevance = True if accept == "accept" else False
            )
            history = pd.concat(
                (
                    history, 
                    pd.DataFrame([new], index = [recommendations.index[index]])
                )
            )
        else:
            history.iloc[index, -1] = True if accept == "accept" else False
    # print(history)
    return history.to_json()


@app.callback(
    Output("store_sentence_embeddings", "data"),
    Output("store_query_embedding", "data"),
    Input("submit", "n_clicks"),
    State("store_sentences", "data"),
    State("query", "value"),
    prevent_initial_call = True
)
def compute_embeddings(clicks, sentences, query):
    sentences = pd.read_json(sentences)
    model = SentenceTransformer(
        'all-MiniLM-L6-v2', 
        cache_folder = "sbert_cache"
    )
    sentence_embeddings = model.encode(
        sentences.text, 
        normalize_embeddings = True,
        batch_size = 4
    ).tolist()
    query_embedding = model.encode(
        [query], 
        normalize_embeddings = True
    ).tolist()
    #vectorizer = CountVectorizer(analyzer = "char", ngram_range = (2, 3))

    #sentence_embeddings = vectorizer.fit_transform(sentences.text).toarray()
    #sentence_embeddings = normalize(sentence_embeddings).tolist()

    #query_embedding = vectorizer.transform([query]).toarray()
    #query_embedding = normalize(query_embedding).tolist()
    return json.dumps(sentence_embeddings), json.dumps(query_embedding)


@app.callback(
    Output("store_recommendations", "data"),
    Input("store_history", "data"),
    Input("store_sentence_embeddings", "data"),
    Input("store_query_embedding", "data"),
    State("store_sentences", "data"),
    prevent_initial_call = True,
)
def update_recommendations(
    history, 
    sentence_embeddings, 
    query_embedding, 
    sentences
):
    history = pd.read_json(history)
    sentence_embeddings = np.array(json.loads(sentence_embeddings))
    query_embedding = np.array(json.loads(query_embedding))
    sentences = pd.read_json(sentences)
    if len(sentences) > 0:
        recommendations = lib.compute_scores(
            sentences, 
            history, 
            sentence_embeddings, 
            query_embedding
        )
        recommendations = recommendations.sort_values(
            "score", 
            ascending = False
        )
    else:
        recommendations = pd.DataFrame()
    return recommendations.to_json()


@app.callback(
    Output("highlighted_words_front", "children"),
    Output("highlighted_lines_front", "children"),
    Output("reviews", "children"),
    Input("store_history", "data"),
)
def update_highlighted(history):
    history = pd.read_json(history)
    relevant = history.query("relevance == True")
    words = len(
        re.findall(r"\w+", " ".join(relevant.text))
    )
    lines = relevant.shape[0]
    reviews = history.shape[0]
    return f"{words:,}", f"{lines:,}", f"{reviews:,}" 


@app.callback(
    Output("accepted_sentences", "children"),
    Input("store_history", "data"),
    prevent_initial_call = True
)
def update_accepted_sentences(history):
    history = pd.read_json(history)
    sentences = history.query("relevance == True").text
    return [html.Li(s) for s in sentences]


@app.callback(
    Output("reviews_query", "children"),
    Output("summary_query", "children"),
    Input("submit", "n_clicks"),
    Input("query", "value"),
    prevent_initial_call = True
)
def update_query_value(clicks, query):
    return query, query


@app.callback(
    Output("barplot", "figure"),
    Output("histogram", "figure"),
    Output("boxplot", "figure"),
    Input("store_recommendations", "data"),
    Input("store_history", "data"),
    State("store_papers", "data"),
    prevent_initial_call = True
)
def update_plots(recommendations, history, papers):
    barplot = None
    histogram = None
    boxplot = None
    if papers:
        papers = pd.read_json(papers)\
        .assign(label = lambda df: df.text.map(lambda x: x[:20] + "..."))\
        .drop(columns = ["text"])

    if history:
        history = pd.read_json(history)\
        .groupby(["paper", "relevance"], dropna = False)\
        .size()\
        .reset_index()\
        .rename(columns = {0:"highlights"})\
        .merge(papers, left_on = "paper", right_on = "paper")\
        .sort_values("paper")\
        .assign(relevance = lambda df: df.relevance.map({1:True, 0:False}))

        barplot = px.histogram(
            data_frame = history, 
            x = "highlights", 
            y = "label", 
            color = "relevance", 
            orientation = "h",
            title = "Highlights by Document",
            labels = {
                "label":"Document", 
                "relevance":"Relevant"
            },
            barmode = "group",
            category_orders = {
                "relevance":[True, False]
            }
        )
        barplot.update_layout(xaxis_title = "Highlights")

    if recommendations:
        recommendations = pd.read_json(recommendations)\
        .merge(papers, left_on = "paper", right_on = "paper")\
        .sort_values("paper")\

        histogram = px.histogram(
            data_frame = recommendations, 
            x = "score", 
            title = "Distribution of the Recommendation Score",
            labels = {"score":"Score"},
            nbins = 30,
            range_x = [0, 1]
        )
        histogram.update_layout(
            # xaxis_range = [0, 1],
            yaxis_title = "Frequency"
        )

        boxplot = px.box(
            data_frame = recommendations, 
            y = "label", 
            x = "score", 
            orientation = "h",
            title = "Distribution of Recommendation Scores by Document",
            labels = {"score":"Score", "label":"Document"},
        )
        boxplot.update_traces(hoverinfo = "skip", hovertemplate = None)
    return barplot, histogram, boxplot


if __name__ == '__main__':
    app.run_server(debug=True, host = "0.0.0.0")
