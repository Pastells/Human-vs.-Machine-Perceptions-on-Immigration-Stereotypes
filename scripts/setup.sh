#!/bin/bash
pip install -r requirements.txt
python3 -m nltk.downloader stopwords
python3 -m spacy download es_core_news_md
