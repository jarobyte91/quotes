import dash
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
import pdftotext as pt
from io import BytesIO
import re
import json
import base64
from sentence_transformers import SentenceTransformer
from nltk.tokenize import PunktSentenceTokenizer
import numpy as np
from scipy.sparse import csr_matrix
from dash.dependencies import Input, Output, State, ALL, MATCH
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from itertools import takewhile
import chardet

import scores
from layout import app, candidates
          
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
    Output("query", "value"),
    Input("add", "n_clicks"),
    Input({"kind":"delete_document", "index":ALL}, "n_clicks"),
    State("upload", "contents"),
    State("store_papers", "data"),
    State("store_sentences", "data"),
    State("upload", "filename"),
    prevent_initial_call = True
    )
def add(
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
    query = ""
    if prop_id == "add.n_clicks" and contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        if filename[-3:] == "pdf":
            file = BytesIO(decoded)
            pdf = pt.PDF(file, raw = True)
            document = "".join(pdf).replace("-\n", "").replace("\n", " ")
            tokenizer = PunktSentenceTokenizer(document)
            sentences = tokenizer.tokenize(document)
        elif filename[-3:] == "txt":
            enc = chardet.detect(decoded)["encoding"]
            document = str(decoded, encoding = enc, errors = "replace")
            tokenizer = PunktSentenceTokenizer(document)
            sentences = tokenizer.tokenize(document)
        else:
            info = json.loads(decoded)
            document = info["document"]
            query = info["query"]
            sentences = document.split("\n\n")
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
        identity = json.loads(prop_id.split(".")[0])
        index = identity["index"]
        kind = identity["kind"]
        if kind == "delete_document" and value:
            filename = store_papers.filename[index] 
            store_papers = store_papers\
                .query(f"filename != '{filename}'")\
                .reset_index(drop = True)
            store_sentences = store_sentences\
                .query(f"filename != '{filename}'")\
                .reset_index(drop = True)
    return store_papers.to_json(), store_sentences.to_json(), query


@app.callback(
    Output("download_txt", "data"),
    Input("download_txt_button", "n_clicks"),
    State("store_history", "data"),
    prevent_initial_call = True
)
def download_txt(clicks, history):
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    content = ""
    filename = "empty"
    if history:
        history = pd.read_json(history)
        content = "\n\n".join(history.query("relevance == True").text)
        filename = "summary.txt"
    return dict(content = content, filename = filename) 


@app.callback(
    Output("download_csv", "data"),
    Input("download_csv_button", "n_clicks"),
    State("store_history", "data"),
    prevent_initial_call = True
)
def download_csv(clicks, history):
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    content = ""
    filename = "empty"
    if history:
        history = pd.read_json(history)
        content = history.to_csv()
        filename = "reviews.csv"
    return dict(content = content, filename = filename) 

@app.callback(
    Output("download_json", "data"),
    Input("download_json_button", "n_clicks"),
    State("store_history", "data"),
    State("store_sentences", "data"),
    State("query", "value"),
    prevent_initial_call = True
)
def download_json(clicks, history, sentences, query):
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    content = ""
    filename = "empty"
    if history and sentences and query:
        filename = "quotes_results.json"
        content = dict(
            sentences = json.loads(sentences),
            query = query,
            history = history
        )
    return dict(content = json.dumps(content), filename = filename) 

@app.callback(
    Output("store_results", "data"),
    Input("store_history", "data"),
    Input("store_sentence_embeddings", "data"),
    Input("store_query_embedding", "data"),
    State("store_sentences", "data"),
)
def update_results(
    history, 
    sentence_embeddings, 
    query_embedding, 
    sentences,
):
    results = pd.DataFrame(
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
            results_index = [i for i in sentences.index if i not in history.index]
            results_embeddings = sentence_embeddings[results_index]
            results = sentences.loc[results_index]
            if isinstance(query_embedding, np.ndarray):
                scores = (query_embedding @ results_embeddings.T).squeeze()
            else:
                scores = (query_embedding @ results_embeddings.T).toarray().squeeze()
            results = results.assign(score = scores)
            results = results.sort_values(
                "score", 
                ascending = False
            )
    return results.to_json()

@app.callback(
    Output("results_content", "children"),
    Input("store_results", "data"),
)
def update_results_body(store_results):
    output = []
    if store_results:
        results = pd.read_json(store_results)
        for i, s in enumerate(results.text.head(candidates)):
            output.append(
                html.Tr(
                    html.Td(
                        s, 
                        id = dict(kind = "results_text", index = i)
                    ),
                    style = dict(background = "lightpink")
                )
            )
    return dbc.Table(output)

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
            recommendations = scores.compute_scores(
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
    Output("suggestions_content", "children"),
    Output("consecutive_strikes", "children"),
    Output("consecutive_strikes", "style"),
    Output("alert", "is_open"),
    Input("store_recommendations", "data"),
    State("store_history", "data"),
    State("alert_checkbox", "checked"),
)
def update_recommendations_body(recommendations, history, alert_checkbox):
    output = []
    strikes_string = ""
    strikes_style = None
    open_alert = False
    if recommendations:
        recommendations = pd.read_json(recommendations)
        for i, s in enumerate(recommendations.text.head(candidates)):
            output.append(
                html.Tr(
                    html.Td(
                        s, 
                        id = dict(kind = "recommendation_text", index = i)
                    ),
                    style = dict(background = "lightpink")
                )
            )
        if history: 
            history = pd.read_json(history)
            strikes = list(
                takewhile(
                    lambda x: not x[1],
                    enumerate(reversed(history.relevance.tolist()), 1)
                )
            )
            if len(strikes) > 0:
                turns = strikes[-1][0] // candidates
            else:
                turns = 0
            strikes_string = f"Turns since last relevant: {turns}"
            if turns < 3:
                strikes_style = {"background":"lightgreen"}
            else:
                strikes_style = {"background":"lightpink"}
                if alert_checkbox:
                    open_alert = True
    return dbc.Table(output), strikes_string, strikes_style, open_alert

@app.callback(
    Output("store_history", "data"),
    Input("clear", "n_clicks"),
    Input("store_sentences", "data"),
    Input("store_query_embedding", "data"),
    Input("store_sentence_embeddings", "data"),
    Input("submit_search", "n_clicks"),
    Input("submit_explore", "n_clicks"),
    Input({"kind":"history_card", "index":ALL}, "n_clicks"),
    # Input({"kind":"document_sentence", "index":ALL}, "n_clicks"),
    State({"kind":"results_text", "index":ALL}, "style"),
    State({"kind":"recommendation_text", "index":ALL}, "style"),
    State("store_history", "data"),
    State("store_recommendations", "data"),
    State("store_results", "data"),
    State("document_dropdown", "value"),
    prevent_initial_call = True
)
def update_history(
    clear,
    store_sentences,
    store_query_embedding,
    store_sentence_embeddings,
    submit_search,
    submit_explore,
    history_card_clicks,
    # document_sentence_clicks,
    results_text_styles,
    recommendation_text_styles,
    store_history,
    store_recommendations,
    store_results,
    dropdown_value
):
    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]
    value = ctx.triggered[0]["value"]
    identity, attribute = prop_id.split(".")
    history = pd.DataFrame(
        columns = ["filename", "sentence", "text", "relevance"]
    )
    if identity not in [
        "clear", 
        "store_sentences",
        "store_query_embedding",
        "store_sentence_embeddings",
    ]:
        recommendations = pd.read_json(store_recommendations)
        results = pd.read_json(store_results)
        history = pd.read_json(store_history)
        sentences = pd.read_json(store_sentences)
        if identity == "submit_search":
            values = [True if x is not None and x["background"] == "lightgreen" else False 
                      for x in results_text_styles]
            new = results[["filename", "sentence", "text"]]\
            .head(candidates)\
            .assign(
                relevance = [x if x is not None else pd.NA for x in values]
            )\
            .set_index(results.index[:candidates])\
            .dropna()
            history = pd.concat(
                (
                    history,
                    new
                ),
            )
        elif identity == "submit_explore":
            values = [True if x is not None and x["background"] == "lightgreen" else False 
                      for x in recommendation_text_styles]
            new = recommendations[["filename", "sentence", "text"]]\
            .head(candidates)\
            .assign(
                relevance = [x if x is not None else pd.NA for x in values]
            )\
            .set_index(recommendations.index[:candidates])\
            .dropna()
            history = pd.concat(
                (
                    history,
                    new
                ),
            )
        else:
            if identity[0] == "{" and identity[-1] == "}":
                identity = json.loads(identity)
                if identity["kind"] == "history_card":
                    new_relevance = [True if (c % 2) == 1 else False 
                                     for c in history_card_clicks]
                    history = history.assign(relevance = new_relevance)
                # elif identity["kind"] == "document_sentence":
                #     index = identity["index"]
                #     in_history = history\
                #     .query(f"filename == '{dropdown_value}' and sentence == {index}")
                #     print("in_history", in_history.shape)
                #     if in_history.shape[0] == 0:
                #         print("new")
                #         original = sentences\
                #         .query(f"filename == '{dropdown_value}'")\
                #         .sort_values("sentence")
                #         new = dict(
                #             filename = dropdown_value,
                #             sentence = index,
                #             text = original.text[index],
                #             relevance = True if value == 1 else False
                #         )
                #         history = pd.concat(
                #             (
                #                 history,
                #                 pd.DataFrame(
                #                     [new], 
                #                     index = [original.index[index]]
                #                 )
                #             )
                #         )
                #     else:
                #         print("append")
                #         position = (history.filename == dropdown_value) & (history.sentence == index)
                #         history.loc[position, "relevance"] = not history.loc[position].relevance.tolist()[0]
    return history.to_json()

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
            card_style = {"background":"lightgreen"} if r["relevance"] else {"background":"lightpink"} 
            row = html.Tr(
                [
                    paper, 
                    sentence, 
                    content, 
                ], 
                id = dict(kind = "history_card", index = i), 
                style = card_style,
                n_clicks = 1 if r["relevance"] else 0
            )
            history_body.append(row)
        return  dbc.Table(history_body)
    else:
        return None

@app.callback(
    Output("store_query_embedding", "data"),
    Input("search", "n_clicks"),
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
    if prop_id == "search.n_clicks" and value and query and vocabulary:
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
    Input("process_documents", "n_clicks"),
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
    if prop_id == "process_documents.n_clicks" and value and sentences:
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
    Output("accepted_sentences", "children"),
    Input("store_history", "data"),
    prevent_initial_call = True
)
def update_accepted_sentences(history):
    history = pd.read_json(history)
    sentences = history.query("relevance == True").text
    return [html.Li(s) for s in sentences]

@app.callback(
    Output("results_query", "children"),
    Input("search", "n_clicks"),
    State("query", "value"),
    prevent_initial_call = True
)
def update_query_value(clicks, query):
    return query

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

        bars = history\
        .assign(relevance = lambda df: df.relevance.map(bool))\
        .groupby(
            ["filename", "relevance"], 
            dropna = False
        )\
        .size()\
        .reset_index()\
        .rename(columns = {0:"highlights"})\
        .merge(papers, left_on = "filename", right_on = "filename")\
        .sort_values("filename")\

        barplot = px.histogram(
            data_frame = bars ,
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
    Input("store_history", "data"),
)
def update_documents_body(sentences, dropdown_value, history):
    ctx = dash.callback_context
    if sentences and history:
        sentences = pd.read_json(sentences)
        history = pd.read_json(history)\
                .query(f"filename == '{dropdown_value}'")
        positive = history.query("relevance == True").sentence.tolist()
        negative = history.query("relevance == False").sentence.tolist()
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
            if i in positive:
                style = {"background":"lightgreen"}
            elif i in negative:
                style = {"background":"lightpink"}
            else: 
                style = {"background":"white"}
            row = html.Tr(
                [
                    html.Td(i + 1),
                    html.Td(r["text"])
                ],
                style = style,
                id = dict(
                    kind = "document_sentence",
                    index = i
                ),
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

@app.callback(
    Output({"kind":"recommendation_text", "index":MATCH}, "style"),
    Input({"kind":"recommendation_text", "index":MATCH}, "n_clicks"),
    prevent_initial_call = True
)
def update_recommendation_colors(clicks):
    ctx = dash.callback_context.triggered[0]
    prop_id = json.loads(ctx["prop_id"].split(".")[0])
    index = prop_id["index"]
    if clicks:
        return {"background":"lightgreen"} if (clicks % 2) == 1 else {"background":"lightpink"}

@app.callback(
    Output({"kind":"results_text", "index":MATCH}, "style"),
    Input({"kind":"results_text", "index":MATCH}, "n_clicks"),
    prevent_initial_call = True
)
def update_results_colors(clicks):
    ctx = dash.callback_context.triggered[0]
    prop_id = json.loads(ctx["prop_id"].split(".")[0])
    index = prop_id["index"]
    if clicks:
        return {"background":"lightgreen"} if (clicks % 2) == 1 else {"background":"lightpink"}

if __name__ == '__main__':
    app.run_server(
        debug = True, 
        host = "0.0.0.0", 
        port = 37639
    )
