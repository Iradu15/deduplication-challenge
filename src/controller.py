from typing import Any
from collections.abc import Hashable
from settings import COLUMNS, LIST_OF_DICT
import numpy as np
import pandas as pd
from collections import defaultdict


class Controller:
    """All methods"""

    @staticmethod
    def assign_ids(df: dict[Hashable, Any]) -> None:
        """Assign row number as value for column 'id'"""
        for k, v in df.items():
            v[COLUMNS.ID.value] = k

    @staticmethod
    def add_additional_columns(df: pd.DataFrame) -> None:
        """Add additional columns to the dataframe"""
        # df[COLUMNS.BASE_ROOT_DOMAIN.value] = df[COLUMNS.ROOT_DOMAIN.value].str.split('.').str[0]
        df[COLUMNS.DETAILS.value] = df.apply(lambda _: {}, axis=1)

    @staticmethod
    def normalize_fields(df: pd.DataFrame) -> None:
        """
        Make fields hashable

        Every field of LIST_OF_DICT is converted from list[dict] to tuple[tuple]
            - [{'original': 'Brown', 'simple': 'Blue'}] => (('original', 'Brown'), ('simple', 'Blue')))
        ENERGY_EFFICIENCY is converted to tuple[tuple] also, but it has different initial format
        ECO_FRIENDLY is converted to list
        It is useful to have PRODUCT_IDENTIFIER as a string instead of tuple
        """
        for field in df:
            df[field] = df[field].apply(lambda x: tuple(x) if isinstance(x, np.ndarray | list) else x)

        for field in LIST_OF_DICT:
            df[field] = df[field].apply(
                lambda x: tuple(tuple(x[i].items()) for i in range(len(x))) if x and isinstance(x, tuple | list) else x
            )

        df[COLUMNS.ENERGY_EFFICIENCY.value] = df[COLUMNS.ENERGY_EFFICIENCY.value].apply(
            lambda x: (tuple(x.items()),) if x and isinstance(x, dict) else x
        )

        df[COLUMNS.ECO_FRIENDLY.value] = df[COLUMNS.ECO_FRIENDLY.value].apply(
            lambda x: [x] if x and not isinstance(x, list) else []
        )

        df[COLUMNS.PRODUCT_IDENTIFIER.value] = df[COLUMNS.PRODUCT_IDENTIFIER.value].apply(
            lambda x: str(x[0]) if isinstance(x, tuple | list) and len(x) == 1 else x
        )

    @staticmethod
    def get_id_to_product_identifier_mapping(df: pd.DataFrame) -> dict[str | tuple, int]:
        """
        Obtain product_identifier to product id (value of the row from the dataset) mapping
        Example:
        15: 'sport wear'
        30: 'clothing'
        """
        df_as_dict = df.to_dict()
        mapping = df_as_dict[COLUMNS.PRODUCT_IDENTIFIER.value]
        del df_as_dict

        return mapping

    @staticmethod
    def filter_products_by_product_id(
        product_identifier_to_product: dict[int, str | tuple], product_identifier: str | tuple
    ) -> list[int]:
        """Get product ids that have 'product_identifier' equal to a specific value"""
        return [k for k, v in product_identifier_to_product.items() if v == product_identifier]

    @staticmethod
    def update_frequencies(field_values: list[Any], frequencies: dict, deduplicated_value: str) -> None:
        """
        All values are decremented, but the most / least frequent one is incremented because of the new merged product.
        Remove keys with empty values
        """
        for value in field_values:
            frequencies[value] -= 1
        frequencies[deduplicated_value] += 1

        frequencies_keys = tuple(frequencies)
        for key in frequencies_keys:
            if not frequencies[key]:
                frequencies.pop(key)

    @staticmethod
    def compute_frequency(df: pd.DataFrame, field: str) -> dict[str, int]:
        """Compute frequency for all values of a field"""
        frequency = defaultdict(int)
        for value in list(df[field]):
            frequency[value] += 1

        return frequency

    @staticmethod
    def compute_values_to_url_mapping(
        field_values: list[Any],
        url_values: list[str],
        values_to_url_mapping: dict,
    ) -> None:
        """Compute values_to_url_mapping needed as argument for add_to_details"""
        for value, url in zip(field_values, url_values):
            values = [value] if isinstance(value, int | str | float | type(None)) else value
            filtered_values = [v for v in values if v is not None]
            for filtered_v in filtered_values:
                v = str(value) if isinstance(value, bool | int | float) else value
                values_to_url_mapping[v].add(url)

    @staticmethod
    def get_all_fields() -> list[str]:
        """Get all fields of a product"""
        return [field.value for field in COLUMNS]

    @staticmethod
    def get_field_values_for_ids(
        df: dict[Hashable, Any],
        ids: list[int],
        field: str,
    ) -> list[Any]:
        """Retrieve the values of a field for products with the specified IDs"""
        return [df.get(id, {}).get(field) for id in ids]

    @staticmethod
    def add_to_details(
        field: str,
        value_to_urls_mapping: dict[str, set],
        deduplicated_product: dict,
        url_special_case: bool = False,
        urls: set[str] = {''},
    ) -> None:
        """
        Add additional value for 'field' to 'details' column
        Example:
        'details' column for a product =
        {
            'product_name': {'name1': {url1, url2}, 'name2': {url3}},
            'unspsc': {'unspsc1': {url2}, 'unspsc2': {url1, url3}},
            'page_url' = {url1, url2}  # this is the special case
        }

        'url_special_case' = True means it retains only the collection, not the mapping.
        This applies to page_url, where we simply want to store all available URLs.
        """
        if not deduplicated_product.get(COLUMNS.DETAILS.value):
            deduplicated_product[COLUMNS.DETAILS.value] = defaultdict(dict)
            deduplicated_product[COLUMNS.DETAILS.value][COLUMNS.PAGE_URL.value] = set()

        if url_special_case:
            deduplicated_product[COLUMNS.DETAILS.value][field].update(urls)
            return

        # update mappings for field
        for k, v in value_to_urls_mapping.items():
            if not deduplicated_product[COLUMNS.DETAILS.value].get(field):
                deduplicated_product[COLUMNS.DETAILS.value][field] = defaultdict(set)
            deduplicated_product[COLUMNS.DETAILS.value][field][k].update(v)

    @staticmethod
    def merge_by_the_most_frequent_value(  # noqa: PLR0913
        df: dict[Hashable, Any],
        products_id: list[int],
        field: str,
        frequencies: dict[str, int],
        deduplicated_product: dict,
        add_to_details: bool = True,
    ) -> None:
        """
        Assign the most frequent value of a field from a group of products to the deduplicated product.

        If `add_to_details` is True, the other available values will be added to the `details` column.

        Example:
        When merging products A and B, if we decide to keep the value from A (`dom1`),
        the `details` column will record that the product is also available at root domain `dom2`.

        Example:
            - A.root_domain = dom1
            - B.root_domain = dom2
            => Result: root_domain = dom1, details = {'root_domain': ['dom2']}
        """
        field_values = Controller.get_field_values_for_ids(df, products_id, field)
        most_frequent_value = max(field_values, key=lambda v: frequencies[v], default='')
        Controller.update_frequencies(field_values, frequencies, most_frequent_value)
        deduplicated_product[field] = most_frequent_value

        if not add_to_details:
            return

        url_values = Controller.get_field_values_for_ids(df, products_id, COLUMNS.PAGE_URL.value)
        values_to_url_mapping = defaultdict(set)

        Controller.compute_values_to_url_mapping(field_values, url_values, values_to_url_mapping)
        Controller.add_to_details(field, values_to_url_mapping, deduplicated_product)

    @staticmethod
    def merge_by_the_least_frequent_value(  # noqa: PLR0913
        df: dict[Hashable, Any],
        products_id: list[int],
        field: str,
        frequencies: dict[str, int],
        deduplicated_product: dict,
        add_to_details: bool = True,
    ) -> None:
        """Assign the least frequent value of a field from a group of products to the deduplicated product."""
        field_values = Controller.get_field_values_for_ids(df, products_id, field)
        least_frequent_value = min(field_values, key=lambda v: frequencies[v], default='')
        Controller.update_frequencies(field_values, frequencies, least_frequent_value)
        deduplicated_product[field] = least_frequent_value

        if not add_to_details:
            return

        url_values = Controller.get_field_values_for_ids(df, products_id, COLUMNS.PAGE_URL.value)
        values_to_url_mapping = defaultdict(set)
        Controller.compute_values_to_url_mapping(field_values, url_values, values_to_url_mapping)
        Controller.add_to_details(field, values_to_url_mapping, deduplicated_product)

    @staticmethod
    def merge_by_the_minimum_value(
        df: dict[Hashable, Any], products_id: list[int], field: str, deduplicated_product: dict
    ) -> None:
        """Assign the minimum value of a field from a group of products to the deduplicated product"""
        deduplicated_product[field] = min(Controller.get_field_values_for_ids(df, products_id, field))

    @staticmethod
    def merge_by_the_lengthiest_value(
        df: dict[Hashable, Any], products_id: list[int], field: str, deduplicated_product: dict
    ) -> None:
        """Assign the lengthiest value of a field from a group of products to the deduplicated product"""
        values = Controller.get_field_values_for_ids(df, products_id, field)
        filtered_values = [v for v in values if v]
        deduplicated_product[field] = max(filtered_values, key=lambda x: len(x), default='')

    @staticmethod
    def merge_url(
        df: dict[Hashable, Any],
        products_id: list[int],
        field: str,
        new_root_domain: str,
        deduplicated_product: dict,
    ) -> None:
        """
        Find the URL that matches the new root domain.

        Example:
        - Old root domain: dom1
        - New root domain: dom2
        The URL should be updated to match the new root domain.
        """
        urls = Controller.get_field_values_for_ids(df, products_id, field)
        urls_as_set = set(urls)

        deduplicated_product[field] = next((url for url in urls if new_root_domain in url), None)

        Controller.add_to_details(field, {}, deduplicated_product, True, urls_as_set)

    @staticmethod
    def merge_by_completing(
        df: dict[Hashable, Any],
        products_id: list[int],
        field: str,
        deduplicated_product: dict,
        add_to_details: bool = True,
    ) -> None:
        """
        Complete the values with unique elements

        Example:
            - intended_industries1 = {A, B}
            - intended_industries2 = {A, C}
            - result = {A, B, C}

        If 'add_to_details' = True it will add to 'details' column that there are multiple available values for this
        specific field
        """
        field_values = Controller.get_field_values_for_ids(df, products_id, field)

        if field == COLUMNS.PRICE.value:
            complete_record = Controller.aggregate_prices(field_values)
        elif field == COLUMNS.SIZE.value:
            complete_record = Controller.aggregate_size(field_values)
        elif field == COLUMNS.ENERGY_EFFICIENCY.value:
            complete_record = Controller.aggregate_energy_efficiency(field_values)
        elif field == COLUMNS.COLOR.value:
            complete_record = Controller.aggregate_color(field_values)
        elif field == COLUMNS.PRODUCTION_CAPACITY.value:
            complete_record = Controller.aggregate_production_capacity(field_values)
        elif field in [COLUMNS.PURITY.value, COLUMNS.PRESSURE_RATING.value, COLUMNS.POWER_RATING.value]:
            complete_record = Controller.aggregate_purity_pressure_rating_power_rating(field_values)
        else:
            complete_record = Controller.compute_complete_record(field_values)

        deduplicated_product[field] = complete_record

        if not add_to_details:
            return

        url_values = Controller.get_field_values_for_ids(df, products_id, COLUMNS.PAGE_URL.value)
        values_to_url_mapping = defaultdict(set)
        Controller.compute_values_to_url_mapping(field_values, url_values, values_to_url_mapping)
        Controller.add_to_details(field, values_to_url_mapping, deduplicated_product)

    @staticmethod
    def compute_complete_record(values: list[Any]) -> set:
        """Compute the complete value of a field containing all the combined values"""
        complete_record = set()
        for value in values:
            if isinstance(value, str | float | int | type(None)):
                complete_record.add(value)
                continue

            complete_record.update(value)

        return complete_record

    @staticmethod
    def aggregate_purity_pressure_rating_power_rating(values: list[tuple]) -> set:
        """
        Aggregate purities & pressure_rating & power_rating (all 3 fields have the same structure)
        into min-max intervals for each dimension and qualitative type.

        Handles cases where the value is a literal (e.g., 'high') instead of a float using
        `literal_value_keys` and `literal_value`. If a literal value is present and no numerical
        value exists for the same key, the literal value will be used.
        """
        result = {}
        literal_value_keys = set()
        literal_values = set()

        for item in values:
            if not item:
                continue

            for entry in item:
                qualitative: str = next(v for k, v in entry if k == 'qualitative')
                unit: str = next(v for k, v in entry if k == 'unit')
                value = next(v for k, v in entry if k == 'value')

                key = (qualitative, unit)

                try:
                    value = float(value)
                except ValueError:
                    literal_value_keys.add(key)
                    literal_values.add(value)
                    continue

                if key not in result:
                    result[key] = {'min': value, 'max': value}
                else:
                    result[key]['min'] = min(result[key]['min'], value)
                    result[key]['max'] = max(result[key]['max'], value)

        result.update({k: {'min': v, 'max': v} for k, v in zip(literal_value_keys, literal_values) if k not in result})

        # Convert to the required output format
        aggregated = [
            (
                ('qualitative', qualitative),
                ('unit', unit),
                ('min', str(values['min'])),
                ('max', str(values['max'])),
            )
            for (qualitative, unit), values in result.items()
        ]

        return set(aggregated)

    @staticmethod
    def aggregate_color(values: list[tuple]) -> set:
        """Aggregate sizes into intervals min-max type for each dimension and measure unit"""
        result = defaultdict(set)

        for item in values:
            if not item:
                continue
            for entry in item:
                original = next(v for k, v in entry if k == 'original')
                simple = next(v for k, v in entry if k == 'simple')

                result[original].add(simple)

        # Convert to required output format
        aggregated = [
            (('original', original), ('simple', ', '.join(sorted(colors)))) for original, colors in result.items()
        ]

        return set(aggregated)

    @staticmethod
    def aggregate_energy_efficiency(values: list[tuple]) -> set:
        """Aggregate energy_efficiency into intervals min-max type for each standard_label and qualitative feature"""
        result = {}

        for item in values:
            if not item:
                continue

            for entry in item:
                qualitative: str = next(v for k, v in entry if k == 'qualitative')
                standard_label: str = next(v for k, v in entry if k == 'standard_label')
                try:
                    value = float(next(v for k, v in entry if k == 'exact_percentage'))
                except TypeError:
                    value = -1.0
                try:
                    max_percentage = float(next(v for k, v in entry if k == 'max_percentage'))
                except TypeError:
                    max_percentage = max(-1.0, value)  # if there is at least one numerical value, take it instead of -1
                try:
                    min_percentage = float(next(v for k, v in entry if k == 'min_percentage'))
                except TypeError:
                    min_percentage = max(
                        -1.0, max_percentage
                    )  # if there is at least one numerical value, take it instead of -1

                key = (qualitative, standard_label)

                if key not in result:
                    result[key] = {
                        'min': min([max_percentage, min_percentage, value]),
                        'max': max([max_percentage, min_percentage, value]),
                    }
                else:
                    result[key]['min'] = min([result[key]['min'], max_percentage, min_percentage, value])
                    result[key]['max'] = max([result[key]['max'], max_percentage, min_percentage, value])

        # Convert to the required output format
        aggregated = [
            (
                ('qualitative', qualitative),
                ('standard_label', standard_label),
                ('min', str(values['min'])),
                ('max', str(values['max'])),
            )
            for (qualitative, standard_label), values in result.items()
        ]

        return set(aggregated)

    @staticmethod
    def aggregate_size(values: list[tuple]) -> set:
        """Aggregate sizes into intervals min-max type for each dimension and measure unit"""
        result = {}
        literal_value_keys = set()
        literal_values = set()

        for item in values:
            if not item:
                continue

            for entry in item:
                dim: str = next(v for k, v in entry if k == 'dimension')
                unit: str = next(v for k, v in entry if k == 'unit')
                value = next(v for k, v in entry if k == 'value')

                key = (dim, unit)

                try:
                    value = float(value)
                except ValueError:
                    literal_value_keys.add(key)
                    literal_values.add(value)
                    continue

                if key not in result:
                    result[key] = {'min': value, 'max': value}
                else:
                    result[key]['min'] = min(result[key]['min'], value)
                    result[key]['max'] = max(result[key]['max'], value)

        result.update({k: {'min': v, 'max': v} for k, v in zip(literal_value_keys, literal_values) if k not in result})

        # Convert to the required output format
        aggregated = [
            (('dimension', str(dim)), ('unit', str(unit)), ('min', str(values['min'])), ('max', str(values['max'])))
            for (dim, unit), values in result.items()
        ]

        return set(aggregated)

    @staticmethod
    def aggregate_production_capacity(values: list[tuple]) -> set:
        """Aggregate production_capacity into intervals min-max type for each (time_frame, unit)"""
        result = {}

        for product_tuple in values:
            if not product_tuple:
                continue

            for items in product_tuple:
                time_frame = next(v for k, v in items if k == 'time_frame')
                unit = next(v for k, v in items if k == 'unit')
                quantity = float(next(v for k, v in items if k == 'quantity'))

                key = (time_frame, unit)

                if key not in result:
                    result[key] = {'min': quantity, 'max': quantity}
                else:
                    result[key]['min'] = min(result[key]['min'], quantity)
                    result[key]['max'] = max(result[key]['max'], quantity)

        return set(
            (('min', str(v['min'])), ('time_frame', time_frame), ('unit', str(unit)), ('max', str(v['max'])))
            for (time_frame, unit), v in result.items()
        )

    @staticmethod
    def aggregate_prices(values: list[tuple]) -> set:
        """Aggregate prices into intervals min-max type for each currency"""
        result = {}

        for product_tuple in values:
            if not product_tuple:
                continue

            for items in product_tuple:
                amount = next(v for k, v in items if k == 'amount')
                currency = next(v for k, v in items if k == 'currency')

                if currency not in result:
                    result[currency] = {'min': amount, 'max': amount}
                else:
                    result[currency]['min'] = min(result[currency]['min'], amount)
                    result[currency]['max'] = max(result[currency]['max'], amount)

        return set((('min', str(v['min'])), ('currency', k), ('max', str(v['max']))) for k, v in result.items())

    @staticmethod
    def convert_to_dataframe(df_as_dict: dict) -> pd.DataFrame:
        """
        Undo normalization such that the dictionary can be converted back to parquet
        Convert tuples to list and nested tuples to list of dictionaries
        """
        for id in list(df_as_dict.keys()):
            for field in LIST_OF_DICT + [COLUMNS.ENERGY_EFFICIENCY.value]:
                value = df_as_dict[id].get(field)
                if value is None:
                    continue
                # print('before', value)
                new_value = []
                for tpl in value:
                    v = {k: v for k, v in tpl}
                    new_value.append(v)

                df_as_dict[id][field] = new_value
                # print('after', new_value)

            if not df_as_dict[id][COLUMNS.DETAILS.value]:
                continue

            for field in [
                COLUMNS.PRODUCTION_CAPACITY.value,
                COLUMNS.PRICE.value,
                COLUMNS.SIZE.value,
                COLUMNS.PURITY.value,
                COLUMNS.PRESSURE_RATING.value,
                COLUMNS.POWER_RATING.value,
                COLUMNS.ENERGY_EFFICIENCY.value,
                COLUMNS.COLOR.value,
            ]:
                value = df_as_dict[id][COLUMNS.DETAILS.value].get(field)
                # print(f'before details {field} -  {value}')
                if value is None:
                    continue
                new_value = []
                for tpl, urls in value.items():
                    v = {k: v for k, v in tpl}
                    v.update({COLUMNS.PAGE_URL.value: urls})
                    new_value.append(v)

                df_as_dict[id][COLUMNS.DETAILS.value][field] = new_value
                # print('after details', new_value)

        df = pd.DataFrame.from_dict(df_as_dict, orient='index')
        df[COLUMNS.PRODUCT_IDENTIFIER.value] = df[COLUMNS.PRODUCT_IDENTIFIER.value].apply(
            lambda x: '' if isinstance(x, tuple) else x
        )
        df[COLUMNS.MANUFACTURING_YEAR.value] = df[COLUMNS.MANUFACTURING_YEAR.value].apply(
            lambda x: [x] if isinstance(x, int) else x
        )
        for field in df:
            df[field] = df[field].apply(lambda x: list(x) if isinstance(x, set) else x)

        return df
