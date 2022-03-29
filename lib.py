from dash import html
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import dash_bootstrap_components as dbc
import numpy as np


def render_document(sentences, history, highlight):
    relevant = history.query("relevance == True").sentence.tolist()
    not_relevant = history.query("relevance != True").sentence.tolist()
    body = []
    for i, s in enumerate(sentences):
        if i == highlight:
            s = html.Strong(s)
        if i in relevant:
            sentence = html.Td(i, style = {"background":"lightgreen"})
            text = html.Td(s, style = {"background":"lightgreen"})
        elif i in not_relevant:
            sentence = html.Td(i, style = {"background":"lightpink"})
            text = html.Td(s, style = {"background":"lightpink"})
        else:
            sentence = html.Td(i)
            text = html.Td(s)
        row = html.Tr([sentence, text])
        if i == highlight:
            row.style = {"outline":"solid", "outline-width":"5px"}
        body.append(row)
    return dbc.Table(body, bordered = True)


def compute_scores(
    sentences, 
    history, 
    sentence_embeddings, 
    query_embedding,
    classifier
):
    relevant = history.query("relevance == True").shape[0]
    not_relevant = history.query("relevance != True").shape[0]
    recommendations_index = [i for i in sentences.index if i not in history.index]
    recommendations_embeddings = sentence_embeddings[recommendations_index]
    recommendations = sentences.loc[recommendations_index]
    if relevant > 0 and not_relevant > 0 and recommendations.shape[0] > 0:
        history_embeddings = sentence_embeddings[history.index]
        if classifier == "Support Vector Machine":
            classifier = SVC(probability = True)
        elif classifier == "Logistic Regression":
            classifier = LogisticRegression()
        elif classifier == "Random Forest":
            classifier = RandomForestClassifier()
        # X = (history_embeddings - query_embedding)**2
        # X = abs(history_embeddings - query_embedding)
        X = history_embeddings
        Y = history.relevance
        classifier.fit(X, Y)
        scores = classifier.predict_proba(recommendations_embeddings)[:, 1]
    else:
        if isinstance(query_embedding, np.ndarray):
            scores = (query_embedding @ recommendations_embeddings.T).squeeze()
        else:
            scores = (query_embedding @ recommendations_embeddings.T).toarray().squeeze()
    recommendations = recommendations.assign(score = scores)
    return recommendations
        
    


