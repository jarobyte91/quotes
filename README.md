# QuOTeS: Query-Oriented Technical Summarization

This is the source code for the paper [QuOTeS: Query-Oriented Technical Summarization](http://export.arxiv.org/abs/2306.11832) by Ramirez-Orta et al., 2023. 

* [Video Tutorial](https://www.youtube.com/watch?v=zR9XisDFQ7w)
* [Online Demo](http://selene.research.cs.dal.ca:37639/)

## Abstract

In the process of writing an academic paper, researchers often spend a lot of time reviewing and summarizing papers to retrieve relevant facts, citations and data to put together the Introduction and Related Work sections of their upcoming research. To address this problem, we propose QuOTeS, an interactive system designed to retrieve sentences relevant to an abstract-like paragraph from a collection of academic papers with the purpose of assisting in the composition of new papers. QuOTeS integrates techniques from Query-Focused Extractive Summarization and Active Learning to provide Interactive Query-Focused Summarization of Scientific Documents. To measure the performance of the system, we performed a comprehensive user study where participants uploaded papers related to their own research and evaluated the system in terms of its usability, its features and the quality of the summaries it produces. Our results show that QuOTeS provides a satisfactory user experience and that it consistently provides query-focused summaries that are relevant, concise and complete.

## Contents

* The main files of the system are **layout.py**, **app.py** and **scores.py**:
  * **layout.py** contains the static elements of the system
  * **app.py** contains the callbacks that make it work
  * **scores.py** contains the code to compute the sentence scores
* The **data** folder contains the raw JSON files collected during the user study of the system.
* The **docker** folder contains the necessary files to build a Docker implementation of the system.
* The **heroku** folder contains the necessary files to upload the Docker implementation to Heroku.

## Installation

Tested on Python 3.8.12

    # QuOTeS requires the Python package pdftotext, which can be cumbersome to install
    # To install it using conda:
    # conda install -c conda-forge pdftotext

    git clone https://github.com/jarobyte91/quotes.git
    cd quotes
    pip install -r requirements.txt
    
    # To run the system:
    python app.py

To build the Docker implementation, the process was split into two steps to make it easier to update:

    # To build the basic dependencies of the system
    sh build_dependencies.sh

    # To build the system itself
    sh build.sh

    # To run the system:
    sh run.sh

## License

This project is license under the MIT License
