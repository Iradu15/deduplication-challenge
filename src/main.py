from enum import Enum
from typing import Any
import numpy as np
import pandas as pd

FILE_PATH = 'db.snappy.parquet'


class COLUMNS(Enum):
    """All columns / features of a product from the sample dataset"""

    UNSPSC = 'unspsc'
    ROOT_DOMAIN = 'root_domain'
    PAGE_URL = 'page_url'
    PRODUCT_TITLE = 'product_title'
    PRODUCT_SUMMARY = 'product_summary'
    PRODUCT_NAME = 'product_name'
    PRODUCT_IDENTIFIER = 'product_identifier'
    BRAND = 'brand'
    INTENDED_INDUSTRIES = 'intended_industries'
    APPLICABILITY = 'applicability'
    ECO_FRIENDLY = 'eco_friendly'
    ETHICAL_AND_SUSTAINABILITY_PRACTICES = 'ethical_and_sustainability_practices'
    PRODUCTION_CAPACITY = 'production_capacity'
    PRICE = 'price'
    MATERIALS = 'materials'
    INGREDIENTS = 'ingredients'
    MANUFACTURING_COUNTRIES = 'manufacturing_countries'
    MANUFACTURING_YEAR = 'manufacturing_year'
    MANUFACTURING_TYPE = 'manufacturing_type'
    CUSTOMIZATION = 'customization'
    PACKAGING_TYPE = 'packaging_type'
    FORM = 'form'
    SIZE = 'size'
    COLOR = 'color'
    PURITY = 'purity'
    ENERGY_EFFICIENCY = 'energy_efficiency'
    PRESSURE_RATING = 'pressure_rating'
    POWER_RATING = 'power_rating'
    QUALITY_STANDARDS_AND_CERTIFICATIONS = 'quality_standards_and_certifications'
    MISCELLANEOUS_FEATURES = 'miscellaneous_features'
    DESCRIPTION = 'description'


class Utils:
    """Methods used once."""

    duplicate_product_identifiers = [
        'Part_Number: 53085',
        'UPC: 705591320215',
        'Part_Number: D966495',
        'Part_Number: DMT206',
        'UPC: 705591195950',
        'SKU: 1783FXDMIDL',
        'SKU: 109139',
        'SKU: Not Available',
        'Part_Number: F1200BT',
        'Part_Number: REU-OADA-C16Z10-B770',
        'SKU: CMP-CNVRT-HDMI2VGA-WHT-KIT',
        'Part_Number: F400B4F',
        'Part_Number: F3200S',
        'Part_Number: D966555',
        'Part_Number: F400S4T-V',
        'Part_Number: 52943',
        'Part_Number: F400B4',
        'Part_Number: 52940',
        'Part_Number: 8784-06',
        'Part_Number: F400BT',
        'Part_Number: LAG14112',
        'UPC: 123456789',
        'Part_Number: F600SF',
        'Part_Number: 56000',
        'Part_Number: F400BT-V',
        'Part_Number: 30440',
        'Part_Number: F400B4-V',
        'Part_Number: F1200B',
        'Part_Number: 32767',
        'CAS: 793-24-8',
        'SKU: 1783SUDCOMK',
        'Part_Number: F400B-V',
        'Part_Number: 52170',
        'Part_Number: HP108',
        'Part_Number: IHB21K6',
        'Part_Number: ZKAD-C3M',
        'SKU: BNDL-V0000-2X',
        'Part_Number: MO31/10',
        'Part_Number: F600BT-V',
        'Part_Number: CFP25G/AE25G/X',
        'NSN: 5360-01-312-9909',
        'Part_Number: BSBHHRP-125',
        'Part_Number: IS410',
        'Part_Number: F1600B',
        'Part_Number: 69622',
        'Part_Number: 0986435505',
        'Part_Number: F600B4F-V',
        'Part_Number: 53007',
        'Part_Number: CMP15G/AE15G/CFP15G',
        'Part_Number: D966615',
        'Part_Number: F400B4F-V',
        'Part_Number: 40350',
        'Part_Number: F600S4',
        'Part_Number: 40149',
        'Part_Number: F400B',
        'Part_Number: 70154BKXS',
        'Part_Number: D966525',
        'SKU: WP1540AW',
        'Part_Number: D966515',
        'Part_Number: FJ2150_NAVY',
        'Part_Number: 90155',
        'Part_Number: MO31/30',
        'Product_Code: B07H9H6SPN',
        'Part_Number: F600B-V',
        'UPC: 705591195868',
        'Part_Number: REU-01075',
        'SKU: 34246',
        'Part_Number: F400B4T',
        'Product_Code: BLD7714-SM',
        'Part_Number: 58773',
        'Product_Code: HT-B8110',
        'CAS: 137-26-8',
        'Part_Number: 66500',
        'SKU: 1783TSYBLD',
        'Product_Code: IHB21K6',
        'Part_Number: REU-98707-2346',
        'Part_Number: F600S4F',
        'Part_Number: F400B-F',
        'SKU: 113129',
        'Part_Number: 29790',
        'Part_Number: 8320BGP',
        'Part_Number: F400SS4',
        'Part_Number: 95848',
        'Part_Number: 7602BKPMED',
        'Part_Number: F600S',
        'Part_Number: D966415',
        'Part_Number: F1200B-V',
        'UPC: 39026',
    ]

    @staticmethod
    def normalize_fields(df: pd.DataFrame):
        """Make fields hashable"""
        for field in df:
            df[field] = df[field].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)

            df[field] = df[field].apply(lambda x: ''.join(str(y) for y in x) if isinstance(x, tuple) and len(x) else x)

            df[field] = df[field].apply(lambda x: str(x) if isinstance(x, dict) else x)

        return df

    @staticmethod
    def get_consistent_fields(group_by_term: str, group: pd.DataFrame) -> set:
        """Get consistent fields in a group"""
        consistent_fields = set()
        for field in group:
            vals = group[field].unique()

            if len(vals) == 1 and vals[0] and vals[0] != -1:
                consistent_fields.add(field)

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
        df = Utils.normalize_fields(df)

        # Group by field1 and check if field2 values are consistent
        grouped = df.groupby(field1)

        consistent_fields_across_all_groups = set({field.value for field in COLUMNS})

        for f1, group in grouped:
            if ignore_empty_fields1 and (not f1 or 'not available' in str(f1).lower()):
                continue

            f2_values = group[field2].unique()

            if len(f2_values) > 1:
                print(f'Discrepancy found for {field1}: {f1}')
                print(f'Different {field2} values: {f2_values}')

                # print consistent fields for products grouped by field1
                consistent_fields_across_all_groups.intersection_update(Utils.get_consistent_fields(field1, group))

        if consistent_across_all_groups and consistent_fields_across_all_groups:
            print(
                f'Consistent fields across all groups:\n    *{"\n    *".join(f for f in consistent_fields_across_all_groups)}\n'
            )

    @staticmethod
    def compare_same_unspsc_smilar_product_identifier():
        """What fields are the same when unspsc are similar / exact and product_identifier is similar: IHB21K6 case"""
        df = pd.read_parquet(
            FILE_PATH,
            filters=[(COLUMNS.BRAND.value, 'in', ['Base Heat On Demand', 'Base Heat'])],
        )

        df[COLUMNS.PRODUCT_IDENTIFIER.value] = df[COLUMNS.PRODUCT_IDENTIFIER.value].apply(
            lambda x: tuple(x) if isinstance(x, np.ndarray) else x
        )

        df = (
            df.groupby(COLUMNS.UNSPSC.value)
            .apply(lambda x: x.sort_values(COLUMNS.PRODUCT_IDENTIFIER.value))
            .reset_index(drop=True)
        )
        df.to_excel('PULA.xlsx')

        # make fields hashable for unique()
        for field in df:
            if field in [COLUMNS.SIZE.value, COLUMNS.COLOR.value]:
                df[field] = df[field].apply(lambda x: ''.join(str(y) for y in x))

            df[field] = df[field].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)

            df[field] = df[field].apply(lambda x: str(x) if isinstance(x, dict) else x)

        for field in df:
            vals = df[field].unique()
            if len(vals) == 1 and vals[0] and vals[0] != -1:
                print('consistent field:', field)

    @staticmethod
    def same_root_domain_and_product_title():
        """Check if products with same root domain and product title can be different"""
        df = pd.read_parquet(
            FILE_PATH,
        )

        # make fields hashable for unique
        df = Utils.normalize_fields(df)

        grouped = df.groupby([COLUMNS.ROOT_DOMAIN.value, COLUMNS.PRODUCT_TITLE.value])

        for (domain, title), group in grouped:
            if len(group) < 2:
                continue
            consistent_field = set()
            for field in group:
                vals = group[field].unique()
                if len(vals) == 1 and vals[0] and vals[0] != -1:
                    consistent_field.add(field)
            if len(consistent_field) == 2:
                print(f'* for {domain} - {title} nothing else is consistent')

    @staticmethod
    def same_url():
        """Answers 'Same url contains > 1 product?'"""
        df = pd.read_parquet(FILE_PATH)
        group = df.groupby(COLUMNS.PAGE_URL.value)
        for url, group in group:
            if len(group) > 1:
                print(f'URL {url} contains > 1 product')


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


if __name__ == '__main__':
    Utils.same_field1_different_field2(COLUMNS.PRODUCT_IDENTIFIER.value, COLUMNS.UNSPSC.value, True, True)
    # Utils.same_field1_different_field2(COLUMNS.PRODUCT_TITLE.value, COLUMNS.UNSPSC.value, True)
    # Utils.same_product_identifier_same_unspfc()
    # Utils.same_product_identifier_same_unspfc()
    # Utils.same_product_title_same_unspfc()
    # Utils.same_root_domain_and_product_title()
    # vals = ReadParquetFile.extract_field(
    #     COLUMNS.ETHICAL_AND_SUSTAINABILITY_PRACTICES.value
    # )

    # ReadParquetFile.save_to_excel()

    # results = ReadParquetFile.extract_field(COLUMNS.PRODUCT_IDENTIFIER.value)
    # print(f'results: {results}')
    # duplicates = ReadParquetFile.get_duplicates_for_field(
    #     COLUMNS.PRODUCT_IDENTIFIER.value
    # )
    # print(f'Duplicates: {duplicates}')

    # parquet_file = pd.read_parquet(
    #     FILE_PATH,
    #     filters=[
    #         (
    #             COLUMNS.UNSPSC.value,
    #             '=',
    #             'Doors',
    #         )
    #     ],
    # )

    # for field in COLUMNS:
    #     if field == COLUMNS.ENERGY_EFFICIENCY:
    #         continue

    #     duplicates = ReadParquetFile.get_duplicates_for_field(field.value)
    #     print(f'Field: {field.value}, Duplicates: {duplicates}')
