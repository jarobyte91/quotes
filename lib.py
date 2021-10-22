from dash import html
import pdftotext as pt
import re
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np
from itertools import zip_longest

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
#             print("span true")
        if span is True and s < threshold:
            ends.append(i)
            span = False
#             print("span false")
    return list(zip_longest(starts, ends, fillvalue = len(scores)))
#     highlights = [{"start": 110, "end":120, "accepted":None}, 
#                       {"start": 1530, "end":1640, "accepted":None},
#                       {"start": 2530, "end":2640, "accepted":None},
#                       {"start": 3530, "end":3640, "accepted":None},
#                       {"start": 4530, "end":4640, "accepted":None},
#                       {"start": 8530, "end":8640, "accepted":None}]
#     return highlights


def compute_scores(tokens, query, window_size):
#     with open("../data/1409.3215.pdf", "rb") as file:
#         pdf = pt.PDF(file, raw = True)
# #     document = "".join(pdf)
#     document = pdf[0]
#     tokens = re.findall(r"[a-zA-Z]+|[^a-zA-Z]", document)
# #     print(len(tokens))
#     query = "In this paper, we propose a novel neural network model called RNN Encoder-Decoder that consists of two recurrent neural networks (RNN). One RNN encodes a sequence of symbols into a fixed-length vector representation, and the other decodes the representation into another sequence of symbols. The encoder and decoder of the proposed model are jointly trained to maximize the conditional probability of a target sequence given a source sequence. The performance of a statistical machine translation system is empirically found to improve by using the conditional probabilities of phrase pairs computed by the RNN Encoder-Decoder as an additional feature in the existing log-linear model. Qualitatively, we show that the proposed model learns a semantically and syntactically meaningful representation of linguistic phrases."

    windows = [tokens[i:i + window_size] for i in range(0, len(tokens) - window_size + 1)]
    windows_df = pd.DataFrame({"tokens":windows, "text":["".join(w) for w in windows]})

    model = TfidfVectorizer(token_pattern = r"[a-zA-Z]+|[^a-zA-Z]")
    window_embeddings = model.fit_transform(windows_df.text)
    query_embedding = model.transform([query])

#     model = SentenceTransformer('all-MiniLM-L6-v2', device = "cuda")
    #model = SentenceTransformer('average_word_embeddings_glove.6B.300d', device = "cuda")
    
    #query_embedding = model.encode(query, convert_to_tensor = True)
#     window_embeddings = model.encode(windows_df.text, convert_to_tensor = True, show_progress_bar = True, batch_size = 128)
#     torch.save(window_embeddings, "window_embeddings.pt")
    #window_embeddings = torch.load("window_embeddings.pt")
#     print("query_embedding", query_embedding.shape)
#     print("window_embeddings", window_embeddings.shape)
#     query_embedding = torch.rand(384)
#     window_embeddings = torch.rand(1104, 384)
#     print("query_embedding", query_embedding.shape)
#     print("window_embeddings", window_embeddings.shape)

    similarities = cosine_similarity(query_embedding, window_embeddings)
#     print(similarities.shape)
    
    windows_df = windows_df.assign(similarity = similarities.flatten().tolist())

    scores = [{"position":i, "token":t, "score":0, "votes":0} for i, t in enumerate(tokens)]

    for i, s in enumerate(windows_df.similarity):
        for d in scores[i:i + window_size]:
            d["score"] += s
            d["votes"] += 1



    scores = pd.DataFrame(scores).assign(avg_score = lambda df: df.score / df.votes)
    #scores.to_pickle("scores.pkl")
        
    
    
    return scores

