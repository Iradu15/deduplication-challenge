from collections import defaultdict
from collections.abc import Hashable
from copy import deepcopy
import json
from math import inf as INF
import numpy as np
import pandas as pd
from typing import Any


from settings import COLUMNS, LIST_OF_DICT


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
        df[COLUMNS.DETAILS.value] = df.apply(lambda _: {}, axis=1)

    @staticmethod
    def normalize_fields(df: pd.DataFrame) -> None:
        """
        Convert fields to hashable types while maintaining consistency.
        Certain fields need to be hashable for grouping products and storing them in a dictionary.

        - Converts every field in LIST_OF_DICT from list[dict] to tuple[tuple].
          Example: [{'original': 'Brown', 'simple': 'Blue'}] -> ((('original', 'Brown'), ('simple', 'Blue')))
        - Converts ENERGY_EFFICIENCY to tuple[tuple] as well, though its initial format differs.
        - Converts ECO_FRIENDLY to a list (no need to be hashable, as all values are -1).
        - Ensures PRODUCT_IDENTIFIER remains a string instead of a tuple for usability.
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
            lambda x: ''.join(x) if isinstance(x, tuple | list) else x
        )

    @staticmethod
    def get_id_to_product_identifier_mapping(df: pd.DataFrame) -> dict[int, str]:
        """
        Obtain product id (value of the row from the dataset) to product_identifier mapping
        Example: {15: '70145EA2', 30: 'UNV10201'}
        """
        df_as_dict = df.to_dict()
        mapping: dict[int, str] = deepcopy(df_as_dict[COLUMNS.PRODUCT_IDENTIFIER.value])
        del df_as_dict

        return mapping

    @staticmethod
    def filter_products_by_product_id(
        product_identifier_to_product: dict[int, str], product_identifier: str
    ) -> list[int]:
        """Retrieve products ID where 'product_identifier' matches a specific value"""
        return [k for k, v in product_identifier_to_product.items() if v == product_identifier]

    @staticmethod
    def update_frequencies(field_values: list[Any], frequencies: dict, deduplicated_value: str) -> None:
        """
        All values are decremented, but the most or least frequent one is incremented because of the new merged product.
        Remove keys with empty values
        """
        for value in field_values:
            frequencies[value] -= 1
        frequencies[deduplicated_value] += 1

        frequencies_keys = list(frequencies.keys())
        for key in frequencies_keys:
            if not frequencies[key]:
                frequencies.pop(key)

    @staticmethod
    def compute_frequency(df: pd.DataFrame, field: str) -> dict[str, int]:
        """Compute frequencies (number of occurrences) for all values of a field"""
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
        """
        Compute values_to_url_mapping needed as argument for add_to_details
        values_to_url_mapping stores values to url mapping:
            - {'Sports Wear': {url1}, 'Athletic Shorts': {url2}}
        """
        for value, url in zip(field_values, url_values):
            values = [value] if isinstance(value, int | str | float | type(None)) else value
            filtered_values = [v for v in values if v is not None]
            for v in filtered_values:
                values_to_url_mapping[str(v) if isinstance(v, int | str | float) else v].add(url)

    @staticmethod
    def get_all_fields() -> list[str]:
        """Get all fields of a product"""
        return [field.value for field in COLUMNS]

    @staticmethod
    def get_field_values_for_ids(
        df: dict[Hashable, Any],
        products_id: list[int],
        field: str,
    ) -> list[Any]:
        """Retrieve the values of a field for products with the specified IDs"""
        return [df.get(id, {}).get(field) for id in products_id]

    @staticmethod
    def add_to_details(
        field: str,
        value_to_urls_mapping: dict[str, set],
        deduplicated_product: dict,
        url_special_case: bool = False,
        urls: set[str] = {''},
    ) -> None:
        """
        Add additional values for 'field' to the 'details' column.

        Example:
        The 'details' column for a product should look like:
        {
            'product_name': {'name1': {url1, url2}, 'name2': {url3}},
            'unspsc': {'unspsc1': {url2}, 'unspsc2': {url1, url3}},
            'page_url': {url1, url2}  # Special case: stores URLs directly
            ...
        }

        If 'url_special_case' is set to True, only the collection of URLs is retained, without the key-value mapping.
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
            => Result: root_domain = dom1, details = {'root_domain': 'dom1': {A.url}, 'dom2': {B.url}}
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
        Completes the values with unique elements.
        For specific fields (handled by independent `if` branches), values are aggregated within their respective
        min-max intervals for each key. For all other fields, it simply appends all available options.

        ### Example:
        - `intended_industries` for `product1` = {A, B}
        - `intended_industries` for `product2` = {A, C}
        - `intended_industries` for the deduplicated product = {A, B, C}

        If `add_to_details = True`, all available values for the specific field are added to the `details` column,
        mapping each value to its corresponding URL.
        """
        field_values = Controller.get_field_values_for_ids(df, products_id, field)

        if field == COLUMNS.PRICE.value:
            complete_record = Controller.aggregate_into_min_max_intervals(field_values, ['currency'], 'amount')
        elif field == COLUMNS.SIZE.value:
            complete_record = Controller.aggregate_into_min_max_intervals(field_values, ['dimension', 'unit'], 'value')
        elif field == COLUMNS.ENERGY_EFFICIENCY.value:
            complete_record = Controller.aggregate_energy_efficiency(field_values)
        elif field == COLUMNS.COLOR.value:
            complete_record = Controller.aggregate_color(field_values)
        elif field == COLUMNS.PRODUCTION_CAPACITY.value:
            complete_record = Controller.aggregate_into_min_max_intervals(
                field_values, ['time_frame', 'unit'], 'quantity'
            )
        elif field in [COLUMNS.PURITY.value, COLUMNS.PRESSURE_RATING.value, COLUMNS.POWER_RATING.value]:
            complete_record = Controller.aggregate_into_min_max_intervals(
                field_values, ['qualitative', 'unit'], 'value'
            )

        else:
            complete_record = Controller.compute_general_complete_record(field_values)

        deduplicated_product[field] = complete_record

        if not add_to_details:
            return

        url_values = Controller.get_field_values_for_ids(df, products_id, COLUMNS.PAGE_URL.value)
        values_to_url_mapping = defaultdict(set)
        Controller.compute_values_to_url_mapping(field_values, url_values, values_to_url_mapping)
        Controller.add_to_details(field, values_to_url_mapping, deduplicated_product)

    @staticmethod
    def compute_general_complete_record(values: list[Any]) -> set:
        """Compute the aggregated value of a field containing all the available values"""
        complete_record = set()
        for value in values:
            if isinstance(value, str | float | int | type(None)):
                complete_record.add(value)
            else:
                complete_record.update(value)

        return complete_record

    @staticmethod
    def aggregate_color(values: list[tuple]) -> set[tuple]:
        """Aggregate sample colors for each original color"""
        result = defaultdict(set)

        for item in values:
            if not item:
                continue

            for entry in item:
                original = next(v for k, v in entry if k == 'original')
                simple = next(v for k, v in entry if k == 'simple')

                result[original].add(simple)

        aggregated = [
            (('original', original), ('simple', ', '.join(sorted(colors)))) for original, colors in result.items()
        ]

        return set(aggregated)

    @staticmethod
    def aggregate_energy_efficiency(values: list[tuple]) -> set[tuple]:
        """
        Aggregate energy_efficiency into min-max intervals for each standard_label and qualitative feature
        In case there are no available values for min & max, they will be assigned -1
        """
        result = {}

        for item in values:
            if not item:
                continue

            for entry in item:
                qualitative: str = next(v for k, v in entry if k == 'qualitative')
                standard_label: str = next(v for k, v in entry if k == 'standard_label')

                max_value = -1.0
                min_value = INF
                for field in ['exact_percentage', 'max_percentage', 'min_percentage']:
                    try:
                        value = float(next(v for k, v in entry if k == field))
                        max_value = max(max_value, value)
                        min_value = min(min_value, value)
                    except TypeError:
                        pass

                key = (qualitative, standard_label)

                if key not in result:
                    result[key] = {
                        'min': min_value if min_value != INF else -1.0,
                        'max': max_value,
                    }
                else:
                    result[key]['min'] = min(result[key]['min'], min_value if min_value != INF else -1.0)
                    result[key]['max'] = max(result[key]['max'], max_value)

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
    def aggregate_into_min_max_intervals(values: list[Any], key_fields: list[str], value_field: str) -> set[tuple]:
        """
        Aggregate into min-max intervals for each key. There are cases when value is literal instead of numerical
        (which are preferred). The literal ones are stored only for keys where numerical is not available.
        min & max are aggregated as strings for consistent format when converting back to parquet in the end

        Args:
            values: List of tuples containing the values to be aggregated.
            key_fields: List of fields used as keys for aggregation.
            value_field: Field used for min-max aggregation.
        """
        result = {}
        literal_keys = set()
        literal_values = set()

        for product_tuple in values:
            if not product_tuple:
                continue

            for items in product_tuple:
                key_fields_as_list = [next(v for k, v in items if k == key_field) for key_field in key_fields]
                key = tuple(key_fields_as_list)
                value = next(v for k, v in items if k == value_field)

                try:
                    value = float(value)
                except ValueError:
                    literal_keys.add(key)
                    literal_values.add(value)
                    continue

                if key not in result:
                    result[key] = {'min': str(value), 'max': str(value)}
                else:
                    result[key]['min'] = str(min(float(result[key]['min']), value))
                    result[key]['max'] = str(max(float(result[key]['max']), value))

        result.update(
            {k: {'min': str(v), 'max': str(v)} for k, v in zip(literal_keys, literal_values) if k not in result}
        )

        keys_tuple = [tuple(zip(key_fields, r_k)) for r_k in result.keys()]
        values_tuple = [tuple(v.items()) for v in list(result.values())]

        return {(*k, *v) for k, v in zip(keys_tuple, values_tuple)}

    @staticmethod
    def convert_to_dataframe(df_as_dict: dict) -> pd.DataFrame:
        """
        Revert normalization to restore close to the original structure for Parquet conversion.

        - Converts tuples back to lists.
        - Converts nested tuples into a list of dictionaries.
        - Only fields in LIST_OF_DICT & ENERGY_EFFICIENCY need these 2 steps from above
        - Converts PRODUCT_IDENTIFIER to string if tuple
        - Converts MANUFACTURING_YEAR to list
        - Converts any set to list
        """
        for _, item in df_as_dict.items():
            for field in LIST_OF_DICT + [COLUMNS.ENERGY_EFFICIENCY.value]:
                values = item.get(field)
                if values is None:
                    continue

                values_as_list = [{k: v for k, v in value} for value in values]
                item[field] = values_as_list

            if not item.get(COLUMNS.DETAILS.value):
                continue

            for field in LIST_OF_DICT + [COLUMNS.ENERGY_EFFICIENCY.value]:
                values = item[COLUMNS.DETAILS.value].get(field)
                if values is None:
                    continue

                # Example of conversion:
                #   - used tuple because set is not JSON serializable
                #   - {(('a', 1), ('b', 2)): ('url1', 'url2')} => [{'a': 1, 'b': 2, 'url': ('url2', 'url1')}]
                new_value = [{**dict(v), COLUMNS.PAGE_URL.value: tuple(urls)} for v, urls in values.items()]
                item[COLUMNS.DETAILS.value][field] = new_value

        df = pd.DataFrame.from_dict(df_as_dict, orient='index')
        df[COLUMNS.PRODUCT_IDENTIFIER.value] = df[COLUMNS.PRODUCT_IDENTIFIER.value].apply(
            lambda x: ''.join(x) if isinstance(x, tuple) else x
        )
        df[COLUMNS.MANUFACTURING_YEAR.value] = df[COLUMNS.MANUFACTURING_YEAR.value].apply(
            lambda x: [x] if isinstance(x, int) else x
        )
        df[COLUMNS.DETAILS.value] = df[COLUMNS.DETAILS.value].apply(lambda x: None if x == {} else x)

        for field in df:
            df[field] = df[field].apply(lambda x: list(x) if isinstance(x, set) else x)

        return df


class StandardizeController:
    """Class for standardizing 'details' feature and various attributes to min-max intervals"""

    @staticmethod
    def standardize_list_dict_fields(df: pd.DataFrame) -> None:
        """
        Convert list[dict] fields to min-max interval constant format because parquet automatically
        aggregates all fields
        """
        for field in [COLUMNS.PURITY.value, COLUMNS.PRESSURE_RATING.value, COLUMNS.POWER_RATING.value]:
            df[field] = df[field].apply(StandardizeController.standardize_purity_pressure_rating_power_rating)

        df[COLUMNS.PRICE.value] = df[COLUMNS.PRICE.value].apply(StandardizeController.standardize_price)

        df[COLUMNS.SIZE.value] = df[COLUMNS.SIZE.value].apply(StandardizeController.standardize_size)

        df[COLUMNS.ENERGY_EFFICIENCY.value] = df[COLUMNS.ENERGY_EFFICIENCY.value].apply(
            StandardizeController.standardize_energy_efficiency
        )

        df[COLUMNS.PRODUCTION_CAPACITY.value] = df[COLUMNS.PRODUCTION_CAPACITY.value].apply(
            StandardizeController.standardize_production_capacity
        )

        df[COLUMNS.DETAILS.value] = df[COLUMNS.DETAILS.value].apply(StandardizeController.standardize_details)

    @staticmethod
    def merge_min_max_intervals(row: Any, key_composed_fields: list[str]) -> Any:
        """
        Merge min-max intervals for each key in the list of dictionaries.
        Treat cases where the value is a literal (eg 'high') instead of a float using `literal_keys` & 'literal_values'

        Example:
            - for price, key is composed of 'currency'
            - [{'min': 1, 'max': 5, 'currency': 'USD'}, {'min': 2, 'max': 6, 'currency': "USD"}] =>
              [{'min': 1, 'max': 6, 'currency': 'USD'}]
        """
        result = {}
        literal_keys = set()
        literal_values = set()

        for item in row:
            if not item:
                continue

            key_as_list = []
            for key in key_composed_fields:
                key_value = item.get(key)
                key_as_list.append(key_value)

            min_v = item.get('min')
            max_v = item.get('max')

            key = tuple(key_as_list)

            try:
                min_v = float(min_v)
                max_v = float(max_v)
            except ValueError:
                literal_keys.add(key)
                literal_values.add((min_v, max_v))
                continue

            if key not in result:
                result[key] = {'min': min_v, 'max': max_v}
            else:
                result[key]['min'] = min(result[key]['min'], min_v)
                result[key]['max'] = max(result[key]['max'], max_v)

        result.update(
            {
                k: {'min': min_v, 'max': max_v}
                for k, (min_v, max_v) in zip(literal_keys, literal_values)
                if k not in result
            }
        )

        return [
            {
                'min': str(v['min']),
                'max': str(v['max']),
                **{key_k: key_v for key_k, key_v in zip(key_composed_fields, key)},
            }
            for key, v in result.items()
        ]

    @staticmethod
    def convert_to_min_max(row: Any, value_key: str) -> Any:
        """
        Helper function to convert a given key to min-max format for consistency.

        Example:
            Initial format:
            - [{'amount': '12.3', 'type': 'exact', 'qualitative': 'True', 'unit': 'None'}]
            Consistent format:
            - [{'min': '12.3', 'max': '12.3', 'qualitative': 'True', 'unit': 'None'}]
        """
        if not row or not isinstance(row, list):
            return []

        for item in row:
            if item.get(value_key):
                item['min'] = item['max'] = str(item.pop(value_key))
                item.pop('type', None)

        return row

    @staticmethod
    def standardize_purity_pressure_rating_power_rating(row: Any) -> Any:
        """
        Standardize purity, pressure_rating and power_rating to min-max intervals.
        All 3 fields have the same structure.
        """
        row = StandardizeController.convert_to_min_max(row, 'value')
        return StandardizeController.merge_min_max_intervals(row, ['qualitative', 'unit'])

    @staticmethod
    def standardize_price(row: Any) -> Any:
        """Standardize prices to min-max intervals."""
        row = StandardizeController.convert_to_min_max(row, 'amount')
        return StandardizeController.merge_min_max_intervals(row, ['currency'])

    @staticmethod
    def standardize_size(row: Any) -> Any:
        """Standardize sizes to min-max intervals."""
        row = StandardizeController.convert_to_min_max(row, 'value')
        return StandardizeController.merge_min_max_intervals(row, ['dimension', 'unit'])

    @staticmethod
    def standardize_energy_efficiency(row: Any) -> Any:
        """Standardize energy efficiency to min-max intervals."""
        if not row or not isinstance(row, list):
            return []

        for item in row:
            if any(field in item for field in ['exact_percentage', 'max_percentage', 'min_percentage']):
                item['min'] = str(item.get('min_percentage', item.get('exact_percentage', '-1')))
                item['max'] = str(item.get('max_percentage', item.get('exact_percentage', '-1')))

                for key in ['exact_percentage', 'min_percentage', 'max_percentage']:
                    item.pop(key, None)

        return row

    @staticmethod
    def standardize_production_capacity(row: Any) -> Any:
        """Standardize production capacity to min-max intervals."""
        row = StandardizeController.convert_to_min_max(row, 'quantity')
        return StandardizeController.merge_min_max_intervals(row, ['time_frame', 'unit'])

    @staticmethod
    def standardize_details(row: Any) -> json:
        """Serialize 'details' field to JSON because parquet automatically aggregates all keys for a field."""
        if not row:
            return json.dumps({})

        normalized_value = {}
        for k, v in row.items():
            if k == COLUMNS.PAGE_URL.value:
                normalized_value[k] = list(v)
            elif k in LIST_OF_DICT + [COLUMNS.ENERGY_EFFICIENCY.value]:
                normalized_value[k] = v
            else:
                normalized_value[k] = {k2: list(v2) for k2, v2 in v.items()}

        return json.dumps(normalized_value)
