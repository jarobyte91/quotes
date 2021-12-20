from dash import html
import pandas as pd
from sklearn.svm import SVC
import dash_bootstrap_components as dbc


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


def compute_scores_dl(sentences, history, sentence_embeddings, query_embedding):
    relevant = history.query("relevance == True").shape[0]
    not_relevant = history.query("relevance != True").shape[0]
    recommendations = [(i, s) for i, s in enumerate(sentences) 
                       if i not in history.sentence.tolist()]
    recommendations = pd.DataFrame(recommendations, 
                                   columns = ["sentence", "text"])
    history_sentences = history.sentence.map(int).values
    recommendations_embeddings = sentence_embeddings[recommendations.sentence.map(int).values]
    if relevant > 0 and not_relevant > 0:
        classifier = SVC(probability = True)
        X = sentence_embeddings[history_sentences]
        Y = history.relevance
        classifier.fit(X, Y)
        scores = classifier.predict_proba(recommendations_embeddings)[:, 1]
    else:
        scores = (query_embedding @ recommendations_embeddings.T).squeeze()
    recommendations = recommendations.assign(score = scores)
    return recommendations
        
    


