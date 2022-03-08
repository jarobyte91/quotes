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
from scipy.sparse import csr_matrix
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
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
# Settings Modal
###################################

embeddings_dropdown = dcc.Dropdown(
    options = [
        {"label":"Word Unigrams", "value":"Word Unigrams"},
        {
            "label":"Character Trigrams", 
            "value":"Character Trigrams"
        }, 
        {
            "label":"Sentence-BERT Embeddings", 
            "value":"Sentence-BERT Embeddings"
        },
    ],
    value = "Word Unigrams",
    id = "embeddings_dropdown",
    clearable = False
)

classifier_dropdown = dcc.Dropdown(
    options = [
        {
            "label":"Logistic Regression", 
            "value":"Logistic Regression", 
        },       
        {
            "label":"Random Forest", 
            "value":"Random Forest", 
        },
        {
            "label":"Support Vector Machine", 
            "value":"Support Vector Machine", 
        },
    ],
    value = "Random Forest",
    id = "classifier_dropdown",
    clearable = False
)

settings_modal = dbc.Modal(
    id = "settings_modal",
    is_open = False,
    children = [
        dbc.ModalHeader(html.H3("Settings")),
        dbc.ModalBody(
            [
                html.H5("Embeddings"),
                embeddings_dropdown,
                html.H5("Classifier"),
                classifier_dropdown,
            ]
        ),
        dbc.ModalFooter(dbc.Button("Close", id = "close")),
    ]
)

###################################
# Upload tab
###################################

add_paper = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                dbc.Button(dcc.Upload("Upload file", id = "upload"))
            ),
            width = 3,
        ),
        dbc.Col(
            dbc.Card(
                dbc.Button(
                    [
                        "Add Paper ",
                    ], 
                    id = "add_paper"
                )
            ),
            width = 3,
        ),
    ]
)   
tab_upload = dbc.Tab(
    label = "Upload", 
    id = "tab_upload", 
    label_style = {"font-size":"1.5em"},
    children = dbc.Container(
        [
            settings_modal,
            html.P(),
            add_paper,
            html.P(),
            dbc.Col(
                dbc.Row(
                    [
                        "file: ",
                        html.Div(id = "filename")
                    ],
                ),
            ),
            html.P(),
            html.H3("Uploaded Papers"),
            html.Div(id = "paper_list_show"),           
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.Button(
                                "Process Papers", 
                                id = "process_papers"
                            )
                        ),
                        width = 3,
                        align = "center"
                    ),
                    dbc.Col(
                        dbc.Card(dbc.Button("Settings", id = "settings")),
                        width = 3,
                        align = "center"
                    ),
                ]
            ),
            html.P(),
            dbc.Col(
               dbc.Table(
                   [
                       html.Tr(
                           [
                               html.Td("Embeddings Dimensions:"),
                               html.Td(
                                   id = "embeddings_dimensions",
                                   style = {"text-align":"right"}
                               ),
                           ],
                           id = "embeddings_status"
                       ),
                       html.Tr(
                           [
                               html.Td("Vocabulary Tokens:"),
                               html.Td(
                                   id = "vocabulary_tokens",
                                   style = {"text-align":"right"}
                               )
                           ],
                           id = "vocabulary_status"
                       )
                   ]
               ),
               width = 4
            )
        ], 
        fluid = True
    )
)

###################################
# History tab
###################################

tab_history = dbc.Tab(
    label = "History", 
    label_style = {"font-size":"1.5em"},
    children = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.H3("Query")),
                    dbc.Col(),
                    dbc.Col(
                        dbc.Card(
                            dbc.Button("Clear history", id = "clear")
                        ),
                        width = 3
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.Button(
                                "Download .csv", 
                                id = "download_csv"
                            )
                        ),
                        width = 3
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

tab_summary = dbc.Tab(
    label = "Summary",
    label_style = {"font-size":"1.5em"},
    children = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col([html.H3("Query")], width = 3),
                dbc.Col(),
                dbc.Col(
                    dbc.Card(
                        dbc.Button("Download .txt", id = "download_txt")
                    ), 
                    width = 3
                )
            ]
        ),
        html.P(id = "summary_query"),
        html.H3("Accepted Sentences"),
        html.Ul(id = "accepted_sentences")
    ],
    fluid = True
    ), 
)

###################################
# Documents tab
###################################

dropdown = dcc.Dropdown(
    id = "document_dropdown",
)
tab_documents = dbc.Tab(
    label = "Documents",
    label_style = {"font-size":"1.5em"}, 
    children = dbc.Container(
        fluid = True,
        children = [
            dropdown,
            dcc.Store(data = 0, id = "document_id"),
            html.Div(id = "documents_body")
        ]
    )
)

###################################
# Search tab
###################################

query = dbc.Container(
    children = [
        dbc.Row(
            [
                dbc.Col(html.H3("Query")),
                dbc.Col(
                    align = "center",
                    width = 3,
                    id = "embeddings_status_search"
                ),
                dbc.Col(
                    dbc.Card(dbc.Button("Submit", id = "submit")), 
                    width = 3
                ),
            ]
        ),
        dbc.Textarea(
            id = "query", 
            rows = 5
        ),
    ], 
    fluid = True
)


header = html.Tr(
    [
        html.Th("Text"),
        html.Th("Relevant?", colSpan = 2, style = {"width":"10%"})
    ]
)
rows = []
for i in range(5):
    content = html.Td(
        html.Div(id = {"kind":"recommendation_text", "index":i})
    )
    accept = dbc.Button(
        "✔", 
        id = dict(
            type = "recommendation_accept", 
            index = i
        ), 
        style = {"background":"seagreen"}
    )
    reject = dbc.Button(
        "✕", 
        id = dict(
            type = "recommendation_reject", 
            index = i
        ), 
        style = {"background":"firebrick"}
    )
    row = html.Tr(
        [
            content, 
            html.Td(accept), 
            html.Td(reject)
        ], 
    )
    rows.append(row)
recommendations_body = dbc.Container(
    [
        html.H3("Suggested Sentences"),
        dbc.Table([header] + rows)
    ],
    fluid = True
)


tab_search = dbc.Tab(
    label = "Search", 
    label_style = {"font-size":"1.5em"}, 
    children =[
        query,
        recommendations_body,
    ],
)

###################################
# Overview tab
###################################

tab_overview = dbc.Tab(
    label = "Dashboard",
    label_style = {"font-size":"1.5em"},
    children = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id = "general"), width = 6),
                    dbc.Col(dcc.Graph(id = "barplot"), width = 6),
                ],
            ),           
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id = "histogram"), width = 6),
                    dbc.Col(dcc.Graph(id = "boxplot"), width = 6),
                ],
            ),
        ],
        fluid = True,
    )
)

###################################
# Stores
###################################

store_sentences = dcc.Store(id = "store_sentences")
store_sentence_embeddings = dcc.Store(id = "store_sentence_embeddings")
store_query_embedding= dcc.Store(id = "store_query_embedding")
history = pd.DataFrame(
    columns = ["filename", "sentence", "text", "relevance"]
).to_json()
store_history = dcc.Store(
    data = history, 
    id = "store_history"
)
store_recommendations = dcc.Store(id = "store_recommendations")
store_papers = dcc.Store(id = "store_papers")
store_vocabulary = dcc.Store(id = "store_vocabulary")
download = dcc.Download(id = "download")
store = dcc.Loading(
    [
        store_sentences,
        store_sentence_embeddings,
        store_query_embedding,
        store_history,
        store_recommendations,
        store_papers,
        store_vocabulary,
        download
    ],
    type = "dot"
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
                tab_documents, 
                tab_search, 
                tab_history, 
                tab_overview, 
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
    .groupby("filename")["sentence"]\
    .size()\
    .reset_index()
    df = papers.merge(sentences)
    output = []
    if len(df) > 0:
        rows = []
        header = html.Thead(
            [
                html.Th(
                    "File", 
                    style = {"width":"20%", "overflow-wrap":"anywhere"}
                ), 
                html.Th("Text"), 
                html.Th("Characters"), 
                html.Th("Words"),
                html.Th("Sentences"),
            ]
        )
        rows.append(header)
        for i, (filename, text, sentences) in enumerate(df.values):
            words = len(list(re.findall(r"\w+", text)))
            r = html.Tr(
                [
                    html.Td(
                        filename if len(filename) <= 30 else filename[:30] + "...", 
                        style = {"width":"20%", "overflow-wrap":"anywhere"}
                    ),
                    html.Td(text[:200] + "..."), 
                    html.Td(
                        f"{len(text):,}", 
                        style = {"text-align":"right"}
                    ), 
                    html.Td(f'{words:,}', style = {"text-align":"right"}), 
                    html.Td(sentences, style = {"text-align":"right"}),
                    html.Td(
                        dbc.Button(
                            "Delete",
                            id = {"kind":"delete_document", "index":i}
                        )
                    ),
                ]
            )
            rows.append(r)
        output = dbc.Table(rows)
    return output


@app.callback(
    Output("filename", "children"),
    Input("upload", "contents"),
    State("upload", "filename")
) 
def show_filename(contents, filename):
    return filename


@app.callback(
    Output("store_papers", "data"),
    Output("store_sentences", "data"),
    Input("add_paper", "n_clicks"),
    Input({"kind":"delete_document", "index":ALL}, "n_clicks"),
    State("upload", "contents"),
    State("store_papers", "data"),
    State("store_sentences", "data"),
    State("upload", "filename"),
    prevent_initial_call = True
    )
def add_paper(
    clicks, 
    delete, 
    contents, 
    store_papers, 
    store_sentences, 
    filename
):
    if store_papers and store_sentences:
        store_papers = pd.read_json(store_papers)
        store_sentences = pd.read_json(store_sentences)
    else:
        store_papers = pd.DataFrame(columns = ["filename", "text"])
        store_sentences = pd.DataFrame(
            columns = ["filename", "sentence", "text"]
        )
    ctx = dash.callback_context
    trigger = ctx.triggered[0]
    prop_id = trigger["prop_id"]
    value = trigger["value"]
    if prop_id == "add_paper.n_clicks" and contents:
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
            [
                (filename, i, s) 
                for i, s in enumerate(sentences)
            ], 
            columns = ["filename", "sentence", "text"]
        )
        store_sentences = pd.concat(
            (store_sentences, sentences_df), 
            ignore_index = True
        )
    else:
        index_kind = json.loads(prop_id.split(".")[0])
        index = index_kind["index"]
        kind = index_kind["kind"]
        if kind == "delete_document" and value:
            filename = store_papers.filename[index] 
            store_papers = store_papers\
                .query(f"filename != '{filename}'")\
                .reset_index(drop = True)
            store_sentences = store_sentences\
                .query(f"filename != '{filename}'")\
                .reset_index(drop = True)
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
    Output({"kind":"recommendation_text", "index":ALL}, "children"),
    Input("store_recommendations", "data"),
)
def update_recommendations_body(recommendations):
    output = [None for i in range(5)]
    if recommendations:
        recommendations = pd.read_json(recommendations)
        if len(recommendations) > 5:
            output = recommendations.text.head().tolist()
    return output


@app.callback(
    Output("history_body", "children"),
    Input("store_history", "data"),
)
def update_history_body(history):
    if history:
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
            if len(r["filename"]) <= 30:
                paper = html.Td(r["filename"])
            else:
                paper = html.Td(r["filename"][:30] + "...")
            sentence = html.Td(r["sentence"] + 1)
            content = html.Td(r["text"])
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
            row = html.Tr(
                [
                    paper, 
                    sentence, 
                    content, 
                    html.Td(accept), 
                    html.Td(reject)
                ], 
                id = dict(type = "history_card", index = i), 
                style = card_style
            )
            history_body.append(row)
        return  dbc.Table(history_body)
    else:
        return None


@app.callback(
    Output("store_history", "data"),
    Input("clear", "n_clicks"),
    Input("store_sentences", "data"),
    Input("store_query_embedding", "data"),
    Input("store_sentence_embeddings", "data"),
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
    history = pd.DataFrame(
        columns = ["filename", "sentence", "text", "relevance"]
    )
    if index_type not in [
        "clear", 
        "store_sentences",
        "store_query_embedding",
        "store_sentence_embeddings"
    ]:
        index_type = json.loads(index_type)
        index = index_type["index"]
        type = index_type["type"]
        subtype, accept = type.split("_")
        history = args[-2]
        recommendations = args[-1]
        if value is None:
            return history
        else:
            recommendations = pd.read_json(recommendations)
            history = pd.read_json(history)
            if subtype == "history":
                history.iloc[index, -1] = True if accept == "accept" else False
            else:
                r = recommendations.to_dict("records")[index]
                new = dict(
                    filename = r["filename"],
                    sentence = r["sentence"], 
                    text = r["text"], 
                    relevance = True if accept == "accept" else False
                )
                history = pd.concat(
                    (
                        history, 
                        pd.DataFrame(
                            [new], 
                            index = [recommendations.index[index]]
                        )
                    )
                )
    return history.to_json()


@app.callback(
    Output("store_query_embedding", "data"),
    Input("submit", "n_clicks"),
    Input("store_sentence_embeddings", "data"),
    State("query", "value"),
    State("store_vocabulary", "data"),
    State("embeddings_dropdown", "value")
)
def compute_query_embedding(
    clicks, 
    sentence_embeddings, 
    query, 
    vocabulary, 
    embeddings_dropdown
):
    query_embedding = None
    ctx = dash.callback_context
    trigger = ctx.triggered[0]
    prop_id = trigger["prop_id"]
    value = trigger["value"]
    if prop_id == "submit.n_clicks" and value and query and vocabulary:
        vocabulary = json.loads(vocabulary)
        if embeddings_dropdown == "Sentence-BERT Embeddings":
            model = SentenceTransformer(
                'all-MiniLM-L6-v2', 
                cache_folder = "sbert_cache",
                device = "cpu"
            )
            query_embedding = model.encode(
                [query],
                normalize_embeddings = True,
                batch_size = 4
            ).tolist()
        elif embeddings_dropdown == "Character Trigrams":
            vectorizer = CountVectorizer(
                analyzer = "char", 
                ngram_range = (3, 3),
                vocabulary = vocabulary
            )
            query_embedding = vectorizer.transform([query])
            query_embedding = dict(
                data = query_embedding.data.tolist(),
                ind = query_embedding.indices.tolist(),
                indptr = query_embedding.indptr.tolist(),
                shape = query_embedding.shape
            )
        elif embeddings_dropdown == "Word Unigrams":
            vectorizer = CountVectorizer(
                analyzer = "word", 
                ngram_range = (1, 1),
                vocabulary = vocabulary
            )
            query_embedding = vectorizer.transform([query])
            query_embedding = dict(
                data = query_embedding.data.tolist(),
                ind = query_embedding.indices.tolist(),
                indptr = query_embedding.indptr.tolist(),
                shape = query_embedding.shape
            )       
        query_embedding = json.dumps(query_embedding)
    return query_embedding


@app.callback(
    Output("store_sentence_embeddings", "data"),
    Output("store_vocabulary", "data"),
    Input("process_papers", "n_clicks"),
    Input("store_sentences", "data"),
    Input("embeddings_dropdown", "value")
)
def compute_sentence_embeddings(clicks, sentences, embeddings_dropdown):
    sentence_embeddings = None
    vocabulary = None
    ctx = dash.callback_context
    trigger = ctx.triggered[0]
    prop_id = trigger["prop_id"]
    value = trigger["value"]
    if prop_id == "process_papers.n_clicks" and value and sentences:
        sentences = pd.read_json(sentences)
        if embeddings_dropdown == "Sentence-BERT Embeddings":
            model = SentenceTransformer(
                'all-MiniLM-L6-v2', 
                cache_folder = "sbert_cache",
                device = "cpu"
            )
            sentence_embeddings = model.encode(
                sentences.text, 
                normalize_embeddings = True,
                batch_size = 4
            ).tolist()
        elif embeddings_dropdown == "Character Trigrams":
            vectorizer = TfidfVectorizer(
                analyzer = "char", 
                ngram_range = (3, 3)
            )
            sentence_embeddings = vectorizer.fit_transform(sentences.text)
            sentence_embeddings = dict(
                data = sentence_embeddings.data.tolist(),
                ind = sentence_embeddings.indices.tolist(),
                indptr = sentence_embeddings.indptr.tolist(),
                shape = sentence_embeddings.shape
            )
            vocabulary = vectorizer.vocabulary_
        elif embeddings_dropdown == "Word Unigrams":
            vectorizer = TfidfVectorizer(
                analyzer = "word", 
                ngram_range = (1, 1)
            )
            sentence_embeddings = vectorizer.fit_transform(sentences.text)
            sentence_embeddings = dict(
                data = sentence_embeddings.data.tolist(),
                ind = sentence_embeddings.indices.tolist(),
                indptr = sentence_embeddings.indptr.tolist(),
                shape = sentence_embeddings.shape
            )          
            vocabulary = vectorizer.vocabulary_
        sentence_embeddings = json.dumps(sentence_embeddings)
        vocabulary = json.dumps(vocabulary)
    return sentence_embeddings, vocabulary


@app.callback(
    Output("store_recommendations", "data"),
    Input("store_history", "data"),
    Input("store_sentence_embeddings", "data"),
    Input("store_query_embedding", "data"),
    Input("classifier_dropdown", "value"),
    State("store_sentences", "data"),
    State("classifier_dropdown", "value"),
)
def update_recommendations(
    history, 
    sentence_embeddings, 
    query_embedding, 
    classifier,
    sentences,
    classifier_state
):
    recommendations = pd.DataFrame(
        columns = ["filename", "sentence", "text", "score"]
    )
    if history and sentence_embeddings and query_embedding and sentences:
        history = pd.read_json(history)
        sentence_embeddings = json.loads(sentence_embeddings)
        query_embedding = json.loads(query_embedding)
        if isinstance(sentence_embeddings, dict):
            sentence_embeddings = csr_matrix(
                (
                    sentence_embeddings["data"],
                    sentence_embeddings["ind"],
                    sentence_embeddings["indptr"]
                ),
                shape = sentence_embeddings["shape"]
            )
            query_embedding = csr_matrix(
                (
                    query_embedding["data"],
                    query_embedding["ind"],
                    query_embedding["indptr"]
                ),
                shape = query_embedding["shape"]
            )
        else:
            sentence_embeddings = np.array(sentence_embeddings)
            query_embedding = np.array(query_embedding)
        sentences = pd.read_json(sentences)
        if len(sentences) > 0:
            recommendations = lib.compute_scores(
                sentences, 
                history, 
                sentence_embeddings, 
                query_embedding,
                classifier
            )
            recommendations = recommendations.sort_values(
                "score", 
                ascending = False
            )
    return recommendations.to_json()


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
    Output("general", "figure"),
    Output("barplot", "figure"),
    Output("histogram", "figure"),
    Output("boxplot", "figure"),
    Input("store_recommendations", "data"),
    Input("store_history", "data"),
    State("store_papers", "data"),
)
def update_plots(recommendations, history, papers):
    general = px.bar(height = 300)
    barplot = px.bar(height = 300)
    histogram = px.bar(height = 300)
    boxplot = px.bar(height = 300)
    if papers and history and recommendations:
        papers = pd.read_json(papers)\
        .assign(label = lambda df: df.text.map(lambda x: x[:20] + "..."))\
        .drop(columns = ["text"])

        history = pd.read_json(history)
        total = history.relevance.size
        relevant = history.relevance.sum()
        not_relevant = total - relevant
        
        general = px.bar(
            x = ["Relevant", "Not Relevant", "Total"],
            color = ["Relevant", "Not Relevant", "Total"],
            y = [relevant, not_relevant, total],
            title = "Sentences",
            labels = {
                "x":"",
                "y":"Sentences",
            },
            height = 300
        )
        general.update_layout(showlegend = False)
        general.update_traces(hoverinfo = "skip", hovertemplate = None)

        barplot = px.histogram(
            data_frame = history\
                .groupby(["filename", "relevance"], dropna = False)\
                .size()\
                .reset_index()\
                .rename(columns = {0:"highlights"})\
                .merge(papers, left_on = "filename", right_on = "filename")\
                .sort_values("filename")\
                .assign(
                    relevance = lambda df: df.relevance.map(
                        {1:True, 0:False}
                    )
                ),
            x = "highlights", 
            y = "label", 
            color = "relevance", 
            orientation = "h",
            title = "Sentences by Document",
            labels = {
                "label":"Document", 
                "relevance":"Relevant"
            },
            barmode = "group",
            category_orders = {
                "relevance":[True, False]
            },
            height = 300
        )
        barplot.update_layout(xaxis_title = "Sentences")
        barplot.update_traces(hoverinfo = "skip", hovertemplate = None)

        recommendations = pd.read_json(recommendations)\
        .merge(papers, left_on = "filename", right_on = "filename")\
        .sort_values("filename")\

        counts, bins = np.histogram(
            recommendations.score, 
            bins = np.arange(0, 1.01, 0.02)
        )
        bins = 0.5 * (bins[:-1] + bins[1:])
        histogram = px.bar(
            x = bins, 
            y = counts,
            title = "Distribution of the Recommendation Score",
            labels = {"x":"Score", "y":"Frequency"},
            height = 300
        )
        histogram.update_layout(
            xaxis_range = [-0.1, 1.1],
            yaxis_title = "Frequency"
        )
        boxplot = px.box(
            data_frame = recommendations, 
            y = "label", 
            x = "score", 
            color = "label",
            orientation = "h",
            title = "Distribution of the Recommendation Score by Document",
            labels = {"score":"Score", "label":"Document"},
            range_x = [-0.1, 1.1],
            color_discrete_sequence = px.colors.qualitative.Pastel,
            height = 300
        )
        boxplot.update_layout(showlegend = False)
        boxplot.update_traces(hoverinfo = "skip", hovertemplate = None)
    return general, barplot, histogram, boxplot


@app.callback(
    Output("documents_body", "children"),
    Input("store_sentences", "data"),
    Input("document_dropdown", "value"),
)
def update_documents_body(sentences, dropdown_value):
    ctx = dash.callback_context
    if sentences:
        sentences = pd.read_json(sentences)
        documents_body = []
        header = html.Thead(
            [
                html.Th("Sentence"),
                html.Th("Text")
            ]
        )
        documents_body.append(header)
        for i, r in enumerate(
            sentences\
                .query(f"filename == '{dropdown_value}'")\
                .to_dict("records")
        ):
            row = html.Tr(
                [
                    html.Td(i + 1),
                    html.Td(r["text"])
                ]
            )
            documents_body.append(row)
        return dbc.Table(documents_body)
    else:
        return None


@app.callback(
    Output("document_dropdown", "options"),
    Input("store_papers", "data"),
    prevent_initial_call = True
)
def update_dropdown_options(papers):
    if papers:
        papers = pd.read_json(papers)
        heads = papers.text.map(lambda x: x[:100] + "...").to_list()
        return [
            {"label":h, "value":f} 
            for h, f in zip(heads, papers.filename)
        ]
    else:
        return None


@app.callback(
    Output("settings_modal", "is_open"),
    Input("settings", "n_clicks"),
    Input("close", "n_clicks"),
    State("settings_modal", "is_open"),
)
def open_settings_modal(settings, close, is_open):
    if settings or close:
        return not is_open
    else:
        return is_open
    

@app.callback(
    Output("embeddings_dimensions", "children"),
    Output("vocabulary_tokens", "children"),
    Output("embeddings_status", "style"),
    Output("embeddings_status_search", "children"),
    Output("embeddings_status_search", "style"),
    Input("store_sentence_embeddings", "data"),
    Input("store_vocabulary", "data"),
)
def update_embeddings_info(sentence_embeddings, vocabulary):
    dims = "None"
    tokens = "None"
    style = {"background":"lightpink"}
    embeddings_status_search = "Sentence Embeddings: Not Ready"
    if sentence_embeddings and vocabulary:
        sentence_embeddings = json.loads(sentence_embeddings)
        if isinstance(sentence_embeddings, dict):
            sentence_embeddings = csr_matrix(
                (
                    sentence_embeddings["data"],
                    sentence_embeddings["ind"],
                    sentence_embeddings["indptr"]
                ),
                shape = sentence_embeddings["shape"]
            )
        else:
            sentence_embeddings = np.array(sentence_embeddings)
        dims = " x ".join([f"{i:,}" for i in sentence_embeddings.shape])
        vocabulary = json.loads(vocabulary)
        if vocabulary:
            tokens = f"{len(vocabulary):,}"
        else:
            tokens = "0"
        style = {"background":"lightgreen"}
        embeddings_status_search = "Sentence Embeddings: Ready"
    return dims, tokens, style, embeddings_status_search, style


if __name__ == '__main__':
    app.run_server(debug=True, host = "0.0.0.0")
