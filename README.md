# Product Deduplication Challenge

## General Description
The objective is to consolidate duplicate product entries into a single, enriched record per product, ensuring uniqueness while maximizing the available information. The dataset consists of product details gathered from multiple web pages using Large Language Models (LLMs), leading to duplicate records where the same product is repeated across different sources. Each record contains partial attributes of a product, and the goal is to intelligently merge these duplicates, enhancing the final product entry with all available details.


## Implementation Details
You can find the details of the final solution and the reasoning behind it by following this: ["**google docs link**"](https://docs.google.com/document/d/1ohn_RBqLT6E_kC_uBGcCMtxU_58cizjHQRsRsCV3tcI/edit?usp=sharing).

## Prerequisites
- install `python3.13`, works as well with `python3.11`.
- run `python3.13 -m venv env`
- go to `./env/bin/activate` script and overwrite the `PYTHONPATH`:
    - `export PYTHONPATH=~/deduplication-challenge/src/` (pwd to `src/`)
- activate env: `source env/bin/activate`
- run `pip instal -r requirements.txt`

## Notes
- run tests using `pytest` from `src/` directory
- wait about `30 seconds` before opening the output file
- `helper.py` contains scripts used at the beginning for dataset research and queries