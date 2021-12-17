from dash import html
import pdftotext as pt
import re
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import torch
import numpy as np
from itertools import zip_longest, accumulate
from nltk.tokenize import PunktSentenceTokenizer
from pprint import pprint

def join(l, sep = ""):
    output = []
    for i in l[:-1]:
        output.append(i)
        output.append(html.Br())
    output.append(l[-1])
    return output

def render_document(t, h):
    if len(t) > 0:
        content = []
        if len(h) > 0:
            content.extend(join("".join(t[:h[0]["start"]]).split("\n")))
            string = "".join(t[h[0]["start"]:h[0]["end"]])
            splitted = string.split("\n")
            splitted = [html.Strong(x, style = {"color":"red"}) for x in splitted]
            content.extend(join(splitted))
            
            for past, current in zip(h[:-1], h[1:]):
                string = "".join(t[past["end"]:current["start"]])
                splitted = string.split("\n")
                content.extend(join(splitted))

                string = "".join(t[current["start"]:current["end"]])
                splitted = string.split("\n")
                splitted = [html.Strong(x, style = {"color":"red"}) for x in splitted]
                content.extend(join(splitted))
            string = "".join(t[h[-1]["end"]:])
            splitted = string.split("\n")
            content.extend(join(splitted))
        else:
            string = "".join(t)
            splitted = string.split("\n")
            content.extend(join(splitted))
    else:
        content = []
    return content

def find_spans(scores, threshold = 0.5):
    span = False
    output = []
    starts = []
    ends = []
    for i, s in enumerate(scores):
        if span is False and s > threshold:
            starts.append(i)
            span = True
        if span is True and s < threshold:
            ends.append(i)
            span = False
    return list(zip_longest(starts, ends, fillvalue = len(scores)))


def compute_scores_word_tfidf(tokens, query, window_size):
    windows = [tokens[i:i + window_size] for i in range(0, len(tokens) - window_size + 1)]
    windows_df = pd.DataFrame({"tokens":windows, "text":["".join(w) for w in windows]})
    model = TfidfVectorizer(token_pattern = r"[a-zA-Z]+|[^a-zA-Z\s]", ngram_range = (1, 2))
    window_embeddings = model.fit_transform(windows_df.text)
    query_embedding = model.transform([query])
    similarities = cosine_similarity(query_embedding, window_embeddings)
    windows_df = windows_df.assign(similarity = similarities.flatten().tolist())
    scores = [{"position":i, "token":t, "score":0, "votes":0} for i, t in enumerate(tokens)]
    for i, s in enumerate(windows_df.similarity):
        for d in scores[i:i + window_size]:
            d["score"] += s
            d["votes"] += 1
    scores = pd.DataFrame(scores).assign(avg_score = lambda df: df.score / df.votes)
    return scores

def compute_scores_word_dl(tokens, query, window_size):
    windows = [tokens[i:i + window_size] for i in range(0, len(tokens) - window_size + 1)]
    windows_df = pd.DataFrame({"tokens":windows, "text":["".join(w) for w in windows]})
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    window_embeddings = model.encode(windows_df.text)
    similarities = cosine_similarity(query_embedding, window_embeddings)
    windows_df = windows_df.assign(similarity = similarities.flatten().tolist())
    scores = [{"position":i, "token":t, "score":0, "votes":0} for i, t in enumerate(tokens)]
    for i, s in enumerate(windows_df.similarity):
        for d in scores[i:i + window_size]:
            d["score"] += s
            d["votes"] += 1
    scores = pd.DataFrame(scores).assign(avg_score = lambda df: df.score / df.votes)
    return scores


def compute_scores_sentence_dl(tokens, query, window_size):
    tokenizer = PunktSentenceTokenizer()
    word_lens = [len(t) for t in tokens]
    offsets = [0] + list(accumulate(word_lens))
    starts = offsets[:-1]
    ends = offsets[1:]
    tokens_extended = list(zip(tokens, starts, ends))
    document = "".join(tokens)
    sentences = list(tokenizer.span_tokenize(document))
    windows = [[t for t, a, b in tokens_extended if a >= current[0] and (b <= next[0] or next[1] == sentences[-1][1])] for current, next in zip(sentences[:-1], sentences[1:])]
    windows_df = pd.DataFrame({"tokens":windows, "text":["".join(w) for w in windows]})
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    window_embeddings = model.encode(windows_df.text)
    similarities = cosine_similarity(query_embedding, window_embeddings)
    windows_df = windows_df.assign(similarity = similarities.flatten().tolist())
    scores = windows_df.explode("tokens")\
    .reset_index(drop = True)\
    .reset_index()\
    .drop(columns = "text")\
    .rename(columns = {"index":"position", "tokens":"token", "similarity":"avg_score"})
    return scores

