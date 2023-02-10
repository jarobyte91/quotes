# QuOTeS: Query-Oriented Technical Summarization

In the process of writing an academic paper, researchers often spend a lot of time reviewing and summarizing papers to retrieve relevant facts, citations and data to put together the Introduction and Related Work sections of their upcoming research. To address this problem, we propose QuOTeS, an interactive system designed to retrieve sentences relevant to an abstract-like paragraph from a collection of academic papers with the purpose of assisting in the composition of new papers. QuOTeS integrates techniques from Query-Focused Extractive Summarization and Active Learning to provide Interactive Query-Focused Summarization of Scientific Documents. To measure the performance of the system, we performed a comprehensive user study where participants uploaded papers related to their own research and evaluated the system in terms of its usability, its features and the quality of the summaries it produces. Our results show that QuOTeS provides a satisfactory user experience and that it consistently provides query-focused summaries that are relevant, concise and complete.

You can watch the walkthrough of the system [here](https://www.youtube.com/watch?v=zR9XisDFQ7w).

You can also try the system [here](http://selene.research.cs.dal.ca:37639/).

## Features

The features that set apart QuOTeS from previous works are the following:

* It receives a short paragraph and a collection of academic documents as input and returns the sentences
relevant to the query from the documents in the collection.
* It is an Interactive Query-Focused Summarization system.
* It is able to extract the text directly from the academic PDFs (and other types of documents) provided by the user at runtime.
* It integrates Active Learning with Relevance sampling in the task of Query-Focused Summarization of Scientific Documents.

## Installation

Tested on Python 3.8.12

    https://github.com/jarobyte91/quotes.git
    cd quotes
    pip install -r requirements.txt
    python app.py

## License

This project is license under the MIT License
