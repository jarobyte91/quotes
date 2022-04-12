import dash
from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
import pandas as pd
import pdftotext as pt
from flask import Flask
 
server = Flask(__name__)
app = dash.Dash(
    __name__, 
    external_stylesheets = [dbc.themes.BOOTSTRAP], 
    server = server, 
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
    value = "Character Trigrams",
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
# Tutorial tab
###################################

tab_tutorial = dbc.Tab(
    label = "Tutorial",
    id = "tab_tutorial",
    disabled = True,
    children = dbc.Container(
        html.Iframe(
            src = "https://www.youtube.com/embed/67Y-A1e8K6U",
            height = 315,
            width = 560
        ),
        fluid = True
    )
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
                        dbc.Card(
                            dbc.Button(
                                "Settings", 
                                id = "settings", 
                                disabled = True
                            )
                        ),
                        width = 3,
                        align = "center"
                    ),
                ]
            ),
            html.P(),
            dbc.Row(
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
                    width = 3
                )
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
                                id = "download_csv_button"
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
    children = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col([html.H3("Query")], width = 3),
                dbc.Col(),
                dbc.Col(
                    dbc.Card(
                        dbc.Button(
                            "Download .txt", 
                            id = "download_txt_button"
                        )
                    ), 
                    width = 3
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.Button(
                            "Download .json", 
                            id = "download_json_button"
                        )
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

# number of sentences presented to the user
candidates = 3

query = dbc.Container(
    children = [
        dbc.Row(
            [
                dbc.Col(html.H3("Query")),
                dbc.Col(
                    width = 3,
                    id = "embeddings_status_search"
                ),
                dbc.Col(
                    dbc.Card(dbc.Button("Submit", id = "submit")), 
                    width = 3
                ),
            ],
            align = "center",
        ),
        dbc.Textarea(
            id = "query", 
            rows = 5
        ),
    ], 
    fluid = True
)

alert = dbc.Alert(
    "You can stop labelling now",
    id = "alert",
    dismissable = True,
    is_open = False,
    color = "warning",
)

recommendations_body = dbc.Container(
    [
       dbc.Row(
            [
                dbc.Col(html.H3("Suggested Sentences")),
                dbc.Col(
                    dbc.Container(
                        id = "consecutive_strikes"
                    ),
                    width = 3,
                ),
            ],
            align = "center",
        ),
        html.Div(id = "suggestions_content"),
        alert,
        dbc.Card(
           dbc.Button(
               "Submit Labels",
               id = "button_submit_labels"
           )
        ),
    ],
    fluid = True
)


tab_search = dbc.Tab(
    label = "Search", 
    children =[
        query,
        html.P(),
        # alert,
        recommendations_body,
    ],
)

###################################
# Dashboard tab
###################################

tab_dashboard = dbc.Tab(
    label = "Dashboard",
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
download_csv = dcc.Download(id = "download_csv")
download_txt = dcc.Download(id = "download_txt")
download_json = dcc.Download(id = "download_json")
store = dbc.Spinner(
    [
        store_sentences,
        store_sentence_embeddings,
        store_query_embedding,
        store_history,
        store_recommendations,
        store_papers,
        store_vocabulary,
        download_csv,
        download_txt,
        download_json
    ],
    fullscreen = True,
    fullscreen_style = {"opacity":0.5},
    spinner_style={"width": "10rem", "height": "10rem"}
)

###################################
# Main Container 
###################################

app.layout = dbc.Container(
    fluid = True, 
    children = [
        html.H1("QuOTeS - Query-Oriented Technical Summarization"),
        store,
        # alert,
        dbc.Tabs(
            [
                tab_tutorial, 
                tab_upload, 
                tab_documents, 
                tab_search, 
                tab_history, 
                tab_dashboard, 
                tab_summary
            ],
            active_tab = "tab-1"
        ),
    ]
)
 
