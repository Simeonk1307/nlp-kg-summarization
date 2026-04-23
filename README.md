# nlp-kg-summarization

This repository is for exploring KGs and how to create them and look into ways to evaluate them better and make summarizations more consistent

There are two main directories
- evaluation: takes the result summaries from src merges them and creates a json for that. gemini is used for running evaluation metrics on this json.
NOTE: need to add GEMINI_API_KEY for using this
- src: does most of the heavy lifting and has summarization scripts which run the models and creates summaries in results directory
- future-work: is some of the informal work that has been done and can be added in later

NOTE: running main.ipynb without loading models will not run u may need to run it as a run-once-before-main.py file

NOTE: you need to have a dataset directory with pubmed_with_triples_v/ train test validation
