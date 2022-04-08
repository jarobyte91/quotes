from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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
        
    


