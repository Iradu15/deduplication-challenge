from collections import defaultdict
from collections.abc import Hashable
import sys
from typing import Any
import pandas as pd


from controller import Controller
from settings import (
    FILE_PATH,
    MERGE_BY_COMPLETING,
    MERGE_BY_LENGTHIEST_VALUE,
    MERGE_BY_MIN_VALUE,
    COLUMNS,
    MERGE_BY_MOST_FREQUENT,
    MERGE_BY_LEAST_FREQUENT,
    OUTPUT_FILE,
)


def merge_group(df: dict[Hashable, Any], products_id: list[int], frequencies: dict[str, dict[str, int]]) -> None:
    """
    Deduplicate a group of products with the same product_identifier by removing incomplete duplicates and adding
    the merged complete product.
    Notes about 'details' field in google docs documentation.
    """
    deduplicated_product = {}

    for field in Controller.get_all_fields():
        # 'unspsc' and 'root_domain' are computed before 'page_url', so they lack the reverse relationship in the
        # 'details' column. It is created after this 'for' loop
        add_to_details = False if field in [COLUMNS.UNSPSC.value, COLUMNS.ROOT_DOMAIN.value] else True

        if field in MERGE_BY_COMPLETING:
            Controller.merge_by_completing(df, products_id, field, deduplicated_product, add_to_details)

        elif field in MERGE_BY_MOST_FREQUENT:
            field_frequencies = frequencies.get(field, {})
            Controller.merge_by_the_most_frequent_value(
                df, products_id, field, field_frequencies, deduplicated_product, add_to_details
            )

        elif field in MERGE_BY_LEAST_FREQUENT:
            field_frequencies = frequencies.get(field, {})
            Controller.merge_by_the_least_frequent_value(
                df, products_id, field, field_frequencies, deduplicated_product, add_to_details
            )

        elif field in MERGE_BY_MIN_VALUE:
            Controller.merge_by_the_minimum_value(df, products_id, field, deduplicated_product)

        elif field in MERGE_BY_LENGTHIEST_VALUE:
            Controller.merge_by_the_lengthiest_value(df, products_id, field, deduplicated_product)

        elif field == COLUMNS.PAGE_URL.value:
            new_root_domain: str = deduplicated_product.get(COLUMNS.ROOT_DOMAIN.value, '')
            Controller.merge_url(df, products_id, field, new_root_domain, deduplicated_product)

    for field in [COLUMNS.UNSPSC.value, COLUMNS.ROOT_DOMAIN.value]:
        field_values = Controller.get_field_values_for_ids(df, products_id, field)
        url_values = Controller.get_field_values_for_ids(df, products_id, COLUMNS.PAGE_URL.value)
        values_to_url_mapping = defaultdict(set)

        Controller.compute_values_to_url_mapping(field_values, url_values, values_to_url_mapping)
        Controller.add_to_details(field, values_to_url_mapping, deduplicated_product)

    # assign the product identifier
    deduplicated_product[COLUMNS.PRODUCT_IDENTIFIER.value] = df[products_id[0]].get(COLUMNS.PRODUCT_IDENTIFIER.value)

    # remove the duplicate products and add the merged & complete one
    for product_id in products_id:
        df.pop(product_id)
    df[deduplicated_product.get(COLUMNS.ID.value)] = deduplicated_product


def merge_by_product_identifier(
    df: dict[Hashable, Any],
    product_identifier_to_product: dict[int, str | tuple],
    frequencies: dict[str, dict[str, int]],
) -> None:
    """
    Group and merge products by 'product_identifier'.
    Only non-empty 'product_identifier' values different from 'SKU: Not Available' are considered.
    """
    product_identifier_values: set[str | tuple] = {
        product_identifier
        for product_identifier in product_identifier_to_product.values()
        if product_identifier and product_identifier != 'SKU: Not Available'
    }

    for product_identifier in product_identifier_values:
        products_id = Controller.filter_products_by_product_id(product_identifier_to_product, product_identifier)
        # only interested in merging groups with at least 2 products
        if len(products_id) < 2:
            continue

        merge_group(df, products_id, frequencies)


def deduplicate(write_file: bool = False) -> None:
    """
    Main method for deduplicating product data.

    Example for 'frequencies':
    {
        'root_domain': {'root1': 15, 'root2': 2},
        'unspsc': {'gardening': 4, 'sport wear': 7},
    }

    'frequencies' is used during merging to select the most / least frequent value for specific fields
    """
    print('Deduplication is starting')

    df = pd.read_parquet(FILE_PATH)
    Controller.add_additional_columns(df)
    Controller.normalize_fields(df)

    frequencies: dict[str, dict[str, int]] = defaultdict(dict)
    for field in MERGE_BY_LEAST_FREQUENT + MERGE_BY_MOST_FREQUENT:
        frequencies[field] = Controller.compute_frequency(df, field)

    product_identifier_to_pid = Controller.get_id_to_product_identifier_mapping(df)

    df_as_dict = df.to_dict(orient='index')
    Controller.assign_ids(df_as_dict)
    del df

    merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)

    if not write_file:
        return

    df = Controller.convert_to_dataframe(df_as_dict)
    df.to_parquet(OUTPUT_FILE)


if __name__ == '__main__':
    print('For writing result to file, pass argument "-p"')
    if (args_count := len(sys.argv)) > 2:
        print(f'One or zero arguments expected, got {args_count - 1}')
        raise SystemExit(2)

    write = True if len(sys.argv) == 2 and sys.argv[1] == '-p' else False

    deduplicate(write)
    print('Task finished')
