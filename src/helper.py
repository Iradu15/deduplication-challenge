from controller import Controller
from collections import defaultdict
from typing import Any
import numpy as np
import pandas as pd

from settings import FILE_PATH, COLUMNS


class Helpers:
    """Used for initial queries and dataset exploring"""

    @staticmethod
    def normalize_fields2(df: pd.DataFrame):
        """Make fields hashable & assign id for each product & complete BASE_ROOT_DOMAIN column"""
        for field in df:
            df[field] = df[field].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)

        # it is useful to have PRODUCT_IDENTIFIER as a string instead of tuple
        df[COLUMNS.PRODUCT_IDENTIFIER.value] = df[COLUMNS.PRODUCT_IDENTIFIER.value].apply(
            lambda x: str(x[0]) if isinstance(x, tuple) and len(x) == 1 else x
        )
        df[field] = df[field].apply(lambda x: ','.join(str(y) for y in x) if isinstance(x, tuple) and len(x) else x)

        df[field] = df[field].apply(lambda x: str(x) if isinstance(x, dict) else x)

        df[field] = df[field].apply(lambda x: ''.join(y for y in x) if isinstance(x, list | tuple) else x)

        return df

    @staticmethod
    def get_consistent_fields(group: pd.DataFrame, f_value: str = '') -> set:
        """Get consistent fields in a group"""
        consistent_fields = set()
        for field in group:
            vals = group[field].unique()

            # if len(vals) == 1 and vals[0] and vals[0] != -1:
            if len(vals) == 1 and ((vals[0] and vals[0] != -1) or field == COLUMNS.PRODUCT_IDENTIFIER.value):
                consistent_fields.add(field)
            elif (
                vals[0]
                and vals[0] != -1
                and vals[0] != [None]
                and field
                not in [
                    COLUMNS.PRODUCT_SUMMARY.value,
                    COLUMNS.MISCELLANEOUS_FEATURES.value,
                    COLUMNS.DESCRIPTION.value,
                ]
            ):
                print(f'field {field}: {group[field].unique()}')

        if consistent_fields:
            print(f'Consistent fields:\n    *{"\n    *".join(f for f in consistent_fields)}\n')

        return consistent_fields

    @staticmethod
    def same_field1_different_field2(
        field1: str, field2: str, consistent_across_all_groups: bool = False, ignore_empty_fields1: bool = False
    ) -> None:
        """
        Answers 'Can any 2 products with same {field1} have different values for {field2}'
        Also prints consistent fields (that are the same for any 2 products) in the group
        consistent_across_all_groups = True => prints the fields that remain constant within all groups
        """
        df = pd.read_parquet(FILE_PATH)
        # make fields hashable for unique()
        df = Helpers.normalize_fields2(df)
        # Group by field1 and check if field2 values are consistent
        grouped = df.groupby(field1)

        consistent_fields_across_all_groups = set({field.value for field in COLUMNS})

        count = 0

        for f1, group in grouped:
            if ignore_empty_fields1 and (not f1 or 'not available' in str(f1).lower() or f1 == [None]):
                continue

            f2_values = group[field2].unique()

            if len(f2_values) <= 1:
                continue

            count += 1

            print(f'Discrepancy found for {field1}: {f1}')
            print(f'Different {field2} values: {f2_values}')

            # print consistent fields for products grouped by field1
            consistent_fields_across_all_groups.intersection_update(Helpers.get_consistent_fields(group, field1))

        if consistent_across_all_groups and consistent_fields_across_all_groups:
            print(
                f'Consistent fields across all groups:\n    *{"\n    *".join(f for f in consistent_fields_across_all_groups)}\n'
            )
        print('COUNT', count)

    @staticmethod
    def same_url() -> None:
        """Answers 'Same url contains > 1 product?'"""
        df = pd.read_parquet(FILE_PATH)
        group = df.groupby(COLUMNS.PAGE_URL.value)
        for url, group in group:
            if len(group) > 1:
                print(f'URL {url} contains > 1 product')

    @staticmethod
    def count_perfect_duplicates() -> None:
        """Count perfect duplicates"""
        df = pd.read_parquet(FILE_PATH)
        df = Controller.normalize_fields(df)

        df = df.groupby(
            COLUMNS.UNSPSC.value,
            group_keys=False,
        ).apply(lambda x: x.sort_values(by=[f.value for f in COLUMNS]))

        prev = None
        duplicates: int = 0

        for _, prod in df.iterrows():
            if prev is not None and prev.equals(prod):
                duplicates += 1
            prev = prod

        print(duplicates)

    @staticmethod
    def same_product_identifier() -> None:
        df = pd.read_parquet(FILE_PATH)
        df = Controller.normalize_fields(df)

        groups = df.groupby(COLUMNS.PRODUCT_IDENTIFIER.value)

        max_len = -1
        duplicate_product_identifiers = set()
        max_group = None
        for product_identifier, group in groups:
            if not product_identifier or 'not available' in str(product_identifier):
                continue

            if len(group) > 1:
                duplicate_product_identifiers.add(product_identifier)

            if max_len < len(group):
                max_len = len(group)
                max_group = product_identifier

        print('duplicate_product_identifiers', duplicate_product_identifiers)
        print(f'Max group: {max_group} with {max_len}')

    @staticmethod
    def consistent_fields_for_duplicate_product_identifiers():
        """Get which are the consistent fields for products with same product_identifiers'"""
        df = pd.read_parquet(FILE_PATH)
        df = Controller.normalize_fields(df)

        grouped = df.groupby(COLUMNS.PRODUCT_IDENTIFIER.value)
        consistent_fields_across_all_groups = set({field.value for field in COLUMNS})

        for f1, group in grouped:
            if not f1 or 'not available' in str(f1).lower() or len(group) <= 1:
                continue

            consistent_fields_across_all_groups.intersection_update(Helpers.get_consistent_fields(group))

        if consistent_fields_across_all_groups:
            print(
                f'Consistent fields across all groups:\n    *{"\n    *".join(f for f in consistent_fields_across_all_groups)}\n'
            )

    @staticmethod
    def group_by_field1_order_by_field2(
        field1: str, fields2: list[str], file_name: str = 'helper.xlsx', df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Group dataset by field1 and order products by field2"""
        if df is None:
            df = pd.read_parquet(FILE_PATH)
            df = Controller.normalize_fields(df)

        # Group and sort within each group
        grouped = df.groupby(field1, group_keys=True).apply(lambda x: x.sort_values(by=fields2))

        # wizardry to maintain group structure
        grouped['group_order'] = grouped[field1].astype('category').cat.codes
        grouped = grouped.sort_values(by=['group_order'] + fields2).drop(columns=['group_order'])

        # OPTION 2 to group
        # # Group and sort within each group
        # grouped = df.groupby(COLUMNS.PRODUCT_TITLE.value, group_keys=False).apply(
        #     lambda x: x.sort_values(by=COLUMNS.UNSPSC.value).reset_index(drop=True)
        # )

        # # Ensure field1 is treated as an ordered category to maintain sorting
        # grouped[COLUMNS.PRODUCT_TITLE.value] = pd.Categorical(
        #     grouped[COLUMNS.PRODUCT_TITLE.value], categories=grouped[COLUMNS.PRODUCT_TITLE.value].unique(), ordered=True
        # )

        # # Sort by the categorical order and fields2
        # grouped = grouped.sort_values(by=[COLUMNS.PRODUCT_TITLE.value, COLUMNS.UNSPSC.value])

        # Keep only the relevant columns
        grouped.to_excel(file_name, index=True)

        return grouped

    @staticmethod
    def check_emtpy_product_identifier():
        df = pd.read_parquet(FILE_PATH)
        df = Controller.normalize_fields(df)
        # df[COLUMNS.INTENDED_INDUSTRIES.value] = df[COLUMNS.INTENDED_INDUSTRIES.value].apply(
        #     lambda x: ','.join(s for s in x) if isinstance(x, tuple) else str(x)
        # )
        count = 0
        for pi in df[COLUMNS.PRODUCT_IDENTIFIER.value]:
            if pi in ('', '()', (), [], None):
                count += 1
        print(count)

    @staticmethod
    def get_empty_product_identifier() -> pd.DataFrame:
        """Get what other fields are constant when product_identifier is empty and grouped by unspsc"""
        df = pd.read_parquet(FILE_PATH)
        df = Controller.normalize_fields(df)
        # filter df for empty PRODUCT_IDENTIFIER
        df = df[
            df[COLUMNS.PRODUCT_IDENTIFIER.value].apply(
                lambda x: isinstance(x, (tuple, list, np.ndarray)) and len(x) == 0
            )
        ]
        return df

    @staticmethod
    def empty_product_identifier_group_by_field(field: str):
        """Get frequency of consistent fields"""
        df = Helpers.get_empty_product_identifier()

        # Group and sort within each group
        grouped = df.groupby(field, group_keys=False).apply(
            lambda x: x.sort_values(by=COLUMNS.UNSPSC.value).reset_index(drop=True)
        )

        # Ensure PRODUCT_TITLE is treated as an ordered category to maintain sorting
        grouped[field] = pd.Categorical(grouped[field], categories=grouped[field].unique(), ordered=True)

        # Sort by the categorical order and fields2
        grouped = grouped.sort_values(by=[field, COLUMNS.UNSPSC.value])

        # Keep only the relevant columns
        grouped.to_excel('PULA2.xlsx', index=True)

        grouped = df.groupby(field)

        freq = {}

        consistent_fields_across_all_groups = set({field.value for field in COLUMNS})

        for unspsc, group in grouped:
            if not unspsc or 'not available' in str(unspsc).lower() or len(group) <= 1:
                continue

            aux = Helpers.get_consistent_fields(group)
            for f in aux:
                if f not in freq:
                    freq[f] = 1
                else:
                    freq[f] += 1

            consistent_fields_across_all_groups.intersection_update(aux)

        if consistent_fields_across_all_groups:
            print(
                f'Consistent fields across all groups:\n    *{"\n    *".join(f for f in consistent_fields_across_all_groups)}\n'
            )
        for k, v in freq.items():
            print(k, '-', v)

    @staticmethod
    def empty_product_identifier_same_field1_same_field2(field1: str, field2: str) -> None:
        df = Helpers.get_empty_product_identifier()
        df = df.groupby(by=[field1, field2])

        freq = {}
        consistent_fields_across_all_groups = set({field.value for field in COLUMNS})

        for (f1, f2), group in df:
            if any(not field or 'not available' in str(field).lower() or len(group) <= 1 for field in [f1, f2]):
                continue

            aux = Helpers.get_consistent_fields(group)
            for f in aux:
                if f not in freq:
                    freq[f] = 1
                else:
                    freq[f] += 1

            consistent_fields_across_all_groups.intersection_update(aux)

        if consistent_fields_across_all_groups:
            print(
                f'Consistent fields across all groups:\n    *{"\n    *".join(f for f in consistent_fields_across_all_groups)}\n'
            )
        for k, v in freq.items():
            print(k, '-', v)

    @staticmethod
    def empty_product_identifier_same_list_of_fields(fields: list[str]) -> None:
        df = Helpers.get_empty_product_identifier()
        df = df.groupby(by=fields)

        freq = {}
        consistent_fields_across_all_groups = set({field.value for field in COLUMNS})
        for *_, group in df:
            if len(group) <= 1 or any(not field or 'not available' in str(field).lower() for field in fields):
                continue

            aux = Helpers.get_consistent_fields(group)
            for f in aux:
                if f not in freq:
                    freq[f] = 1
                else:
                    freq[f] += 1

            consistent_fields_across_all_groups.intersection_update(aux)

        if consistent_fields_across_all_groups:
            print(
                f'Consistent fields across all groups:\n    *{"\n    *".join(f for f in consistent_fields_across_all_groups)}\n'
            )
        for k, v in freq.items():
            print(k, '-', v)

    @staticmethod
    def same_list_of_fields(fields: list[str]) -> None:
        """
        Answers 'what are the consistent fields when any 2 products have the same values for any 2 fields
        in a given list of fields
        """
        df = pd.read_parquet(FILE_PATH)
        df = Controller.normalize_fields(df)
        df = df.groupby(fields)

        freq = defaultdict(int)
        consistent_fields_across_all_groups = set({field.value for field in COLUMNS})
        for *_, group in df:
            if (
                # len(group[COLUMNS.ROOT_DOMAIN.value].unique()) == 1  # WE NEED DIFFERENT ROOT_DOMAIN
                len(group) <= 1  # WE WANT GROUPS CONTAINING >= 2 products
                or any(
                    not list(group[field].unique())
                    or list(group[field].unique()) == [()]
                    or 'not available' in str(list(group[field].unique())).lower()
                    or list(group[field].unique()) is None
                    for field in fields
                )
            ):
                continue

            aux = Helpers.get_consistent_fields(group)
            for f in aux:
                freq[f] += 1

            consistent_fields_across_all_groups.intersection_update(aux)

        if consistent_fields_across_all_groups:
            print(
                f'Consistent fields across all groups:\n    *{"\n    *".join(f for f in consistent_fields_across_all_groups)}\n'
            )
        for k, v in freq.items():
            print(k, '-', v)


class ReadParquetFile:
    """Files related to reading Parquet files."""

    @staticmethod
    def flatten_value(value: Any, uniques: set):
        """Recursively flatten values into a set."""
        if isinstance(value, str | int | float):
            uniques.add(value)
            return

        if isinstance(value, dict):
            uniques.add(str(value))
            return

        if isinstance(value, list | tuple | set):
            for v in value:
                ReadParquetFile.flatten_value(v, uniques)
            return

        if isinstance(value, np.ndarray):
            if value.size > 0:
                for v in value:
                    ReadParquetFile.flatten_value(v, uniques)
            return

    @staticmethod
    def get_duplicates_for_field(field: str, file=FILE_PATH) -> list:
        """Get all the duplicates for a specified field."""
        # Read the Parquet file using pandas
        df = pd.read_parquet(file)

        if field not in df.columns:
            raise ValueError(f"Field '{field}' not found in the parquet file.")

        uniques = set()
        duplicates = []

        for value in df[field].dropna():
            # If the value is a list / tuple / set, flatten it
            if isinstance(value, str | int | float):
                if value in uniques:
                    duplicates.append(value)
                uniques.add(value)

                continue

            try:
                for v in value:
                    if v in uniques:
                        duplicates.append(v)
                    uniques.add(v)
            except TypeError:
                pass

        return list(set(duplicates))

    @staticmethod
    def extract_field(field: str, file: str = FILE_PATH, save: bool = False) -> list:
        """Extract sorted unique values for specified field."""
        # Read the Parquet file using pandas
        df = pd.read_parquet(file)

        if field not in df.columns:
            raise ValueError(f"Field '{field}' not found in the parquet file.")

        uniques = set()

        for value in df[field].dropna():
            ReadParquetFile.flatten_value(value, uniques)

        sorted_results = sorted(uniques)

        if save:
            # Save to Excel
            df = pd.DataFrame(sorted_results, columns=[field])
            df.to_excel(f'{field}.xlsx', index=True)
            print(f"Excel file saved as '{field}.xlsx'")

        return sorted_results

    @staticmethod
    def save_to_excel(name: str = 'db2.xlsx') -> None:
        parquet_file = (
            pd.read_parquet(
                FILE_PATH,
            )
            .groupby(
                COLUMNS.UNSPSC.value,
                group_keys=False,
            )
            .apply(lambda x: x.sort_values(by=COLUMNS.PRODUCT_NAME.value))
        )
        parquet_file.to_excel(name)

    @staticmethod
    def extract_fields():
        parquet_file = pd.read_parquet(FILE_PATH)
        columns = parquet_file.info()
        print('columns', columns)
