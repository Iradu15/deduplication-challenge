# Product Deduplication

## General Description
The objective is to consolidate duplicate product entries into a single, enriched record per product, ensuring uniqueness while maximizing the available information. The dataset consists of product details gathered from multiple web pages using Large Language Models (LLMs), leading to duplicate records where the same product is repeated across different sources. Each record contains partial attributes of a product, and the goal is to intelligently merge these duplicates, enhancing the final product entry with all available details.


## Implementation Details
You can find the details of the final solution and the reasoning behind it by following this: [link](https://trello.com/b/9HMD6CwE/backend](https://docs.google.com/document/d/1ohn_RBqLT6E_kC_uBGcCMtxU_58cizjHQRsRsCV3tcI/edit?usp=sharing).

## Prerequisites
- run `python3.11 -m venv env`
- go to `./env/bin/activate` script overwrite the PYTHONPATH:
    - export PYTHONPATH=~/veridion/src/
- activate env: `source env/bin/activate`
- run `pip instal -r requirements.txt`

