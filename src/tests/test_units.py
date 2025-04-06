from copy import deepcopy
from collections import defaultdict
import pandas as pd
from typing import Literal


from controller import Controller, StandardizeController
from main import merge_group
from settings import (
    MERGE_BY_LENGTHIEST_VALUE,
    MERGE_BY_MIN_VALUE,
    MERGE_BY_MOST_FREQUENT,
    MERGE_BY_LEAST_FREQUENT,
    SAMPLE_FILE_PATH,
    COLUMNS,
)
from tests import constants


class TestHelperMethods:
    """Tests for helper methods"""

    @staticmethod
    def test_frequency_count() -> None:
        """Test that computing frequencies (done via compute_frequency) works as intended"""
        df = pd.DataFrame.from_dict(constants.SAMPLE_PRODUCTS, orient='index')
        frequencies: dict[str, dict[str, int]] = defaultdict(dict)
        for field in MERGE_BY_LEAST_FREQUENT + MERGE_BY_MOST_FREQUENT:
            frequencies[field] = Controller.compute_frequency(df, field)

        frequencies_as_dict = {k: dict(v) for k, v in frequencies.items()}
        assert frequencies_as_dict == constants.COMPUTE_FREQUENCIES_RESULT

    @staticmethod
    def test_product_identifier_to_pid() -> None:
        """Test mapping pid to product_identifier functionality (done via get_id_to_product_identifier_mapping)"""
        df = pd.read_parquet(SAMPLE_FILE_PATH)
        Controller.normalize_fields(df)

        product_identifier_to_pid = Controller.get_id_to_product_identifier_mapping(df)
        assert product_identifier_to_pid == constants.ID_TO_PRODUCT_IDENTIFIER_MAPPING

    @staticmethod
    def test_assign_ids() -> None:
        """Test assigning ids functionality"""
        df_as_dict: dict = constants.SAMPLE_PRODUCTS
        Controller.assign_ids(df_as_dict)
        assert all(k == v.get(COLUMNS.ID.value) for k, v in df_as_dict.items())

    @staticmethod
    def test_filter_products_by_product_id() -> None:
        """Test filtering products by specific product_id works"""
        product_identifier = 'CAS: 137-26-8'
        products_id = Controller.filter_products_by_product_id(
            constants.ID_TO_PRODUCT_IDENTIFIER_MAPPING, product_identifier
        )
        assert products_id == [9971, 10275]


class TestDeduplicateMethods:
    """Tests for deduplication (merging) methods"""

    @staticmethod
    def test_add_to_details() -> None:
        """
        Test that 'values_to_url_mapping' is added successfully to 'details' field
        Interested in checking the added field inside 'details' is initialized and assigned the right value
        """
        values_to_url_mapping = {
            'TMTD (Tetramethylthiuram Disulfide)': {'url_0'},
            'Rubber Accelerator TMTD IPPD': {'url_1'},
            'Rubber Processing': {'url_3', 'url_2'},
            'Agriculture': {'url_2'},
            -1: {'url_5'},
            None: {'url_6'},
            (('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '96.0')): {'url_8', 'url_7'},
            (('qualitative', True), ('type', 'exact'), ('unit', None), ('value', 'high')): {'url_8'},
        }
        field = COLUMNS.PRODUCT_TITLE.value
        deduplicated_product = {}
        Controller.add_to_details(field, values_to_url_mapping, deduplicated_product)

        # unnecessary at this step
        deduplicated_product.get(COLUMNS.DETAILS.value, {}).pop(COLUMNS.PAGE_URL.value)

        expected_result = {
            COLUMNS.DETAILS.value: {
                COLUMNS.PRODUCT_TITLE.value: {
                    'TMTD (Tetramethylthiuram Disulfide)': {'url_0'},
                    'Rubber Accelerator TMTD IPPD': {'url_1'},
                    'Rubber Processing': {'url_3', 'url_2'},
                    'Agriculture': {'url_2'},
                    -1: {'url_5'},
                    None: {'url_6'},
                    (('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '96.0')): {'url_8', 'url_7'},
                    (('qualitative', True), ('type', 'exact'), ('unit', None), ('value', 'high')): {'url_8'},
                }
            }
        }

        assert deduplicated_product == expected_result

    @staticmethod
    def test_compute_values_to_url_mapping_for_merge_by_completing() -> None:
        """
        Test that values to url mapping (via `values_to_url_mapping`) is correctly created for any field
        in `MERGE_BY_COMPLETING`.

        The test covers cases with:
        - Duplicate values within tuples or as standalone values.
        - Tuples that need to be split before processed.
        """
        field_values = [
            'TMTD (Tetramethylthiuram Disulfide)',
            'Rubber Accelerator TMTD IPPD',
            ('Rubber Processing', 'Agriculture'),
            ('Rubber Processing',),
            (),
            -1,
            None,
            ((('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '96.0')),),
            (
                (('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '96.0')),
                (('qualitative', True), ('type', 'exact'), ('unit', None), ('value', 'high')),
            ),
        ]
        url_values = [f'url_{i}' for i in range(len(field_values))]

        values_to_url_mapping = defaultdict(set)
        Controller.compute_values_to_url_mapping(field_values, url_values, values_to_url_mapping)

        expected_result = {
            'TMTD (Tetramethylthiuram Disulfide)': {'url_0'},
            'Rubber Accelerator TMTD IPPD': {'url_1'},
            'Rubber Processing': {'url_3', 'url_2'},
            'Agriculture': {'url_2'},
            '-1': {'url_5'},
            (('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '96.0')): {'url_8', 'url_7'},
            (('qualitative', True), ('type', 'exact'), ('unit', None), ('value', 'high')): {'url_8'},
        }

        assert values_to_url_mapping == expected_result

    @staticmethod
    def test_complete_record() -> None:
        """
        Test that completing a record with all available values work
        Contains each possible type that is passed as argument for compute_general_complete_record: None, int, tuple
        """
        field_values = [
            ('Powder',),
            ('Powder', 'Granules'),
            ('Rubber Processing', 'Agriculture'),
            ('Rubber Industry',),
            None,
            -1,
        ]

        expected_result = {None, 'Granules', 'Rubber Industry', 'Powder', 'Agriculture', 'Rubber Processing', -1}
        assert Controller.compute_general_complete_record(field_values) == expected_result

    @staticmethod
    def test_aggregate_color() -> None:
        """Test that aggregating colors works as expected"""
        field_values = [
            (),
            (
                (('original', 'Midlands'), ('simple', 'White')),
                (('original', 'Midlands'), ('simple', 'Gray')),
            ),
            ((('original', 'Midlands'), ('simple', 'Gray')),),
            ((('original', 'Midlands'), ('simple', 'Blue')),),
            ((('original', 'Orange'), ('simple', 'Blue')),),
        ]

        expected_result = {
            (('original', 'Midlands'), ('simple', 'Blue, Gray, White')),
            (('original', 'Orange'), ('simple', 'Blue')),
        }
        assert Controller.aggregate_color(field_values) == expected_result

    @staticmethod
    def test_aggregate_purity_conflict() -> None:
        """
        Test that aggregating purity works as expected when there is a conflict between
        literal value 'high' and numerical ones. Both reside as value for the same key.
        The test is the same for pressure_rating and power_rating due to their structure
        """
        field_values = [
            ((('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '99.998')),),
            (),
            (
                (('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '90.998')),
                (('qualitative', False), ('type', 'exact'), ('unit', None), ('value', 'high')),
                (('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '89.998')),
            ),
        ]

        expected_result = {
            (('qualitative', False), ('unit', None), ('min', '89.998'), ('max', '99.998')),
        }
        assert (
            Controller.aggregate_into_min_max_intervals(field_values, ['qualitative', 'unit'], 'value')
            == expected_result
        )

    @staticmethod
    def test_aggregate_purity_no_conflict() -> None:
        """
        Test that aggregating purity works as expected when there is no conflict between literal value 'high'
        and numerical ones.
        The test is the same for pressure_rating and power_rating due to their structure
        """
        field_values = [
            ((('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '99.998')),),
            (),
            (
                (('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '90.998')),
                (('qualitative', True), ('type', 'exact'), ('unit', None), ('value', 'high')),
                (('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '89.998')),
            ),
        ]

        expected_result = {
            (('qualitative', True), ('unit', None), ('min', 'high'), ('max', 'high')),
            (('qualitative', False), ('unit', None), ('min', '89.998'), ('max', '99.998')),
        }

        assert (
            Controller.aggregate_into_min_max_intervals(field_values, ['qualitative', 'unit'], 'value')
            == expected_result
        )

    @staticmethod
    def test_aggregate_size() -> None:
        """Test that aggregating sizes by dimension and unit works as expected"""
        field_values = [
            (
                (('dimension', 'Height'), ('qualitative', False), ('type', 'exact'), ('unit', 'in'), ('value', '20.7')),
                (('dimension', 'Width'), ('qualitative', False), ('type', 'exact'), ('unit', 'in'), ('value', '16.9')),
                (('dimension', 'Weight'), ('qualitative', False), ('type', 'exact'), ('unit', 'lbs'), ('value', '190')),
            ),
            (
                (('dimension', 'Height'), ('qualitative', False), ('type', 'exact'), ('unit', 'in'), ('value', '30.7')),
                (('dimension', 'Weight'), ('qualitative', False), ('type', 'exact'), ('unit', 'lbs'), ('value', '120')),
            ),
            ((('dimension', 'Height'), ('qualitative', False), ('type', 'exact'), ('unit', 'cm'), ('value', '209')),),
        ]

        expected_result = {
            (('dimension', 'Weight'), ('unit', 'lbs'), ('min', '120.0'), ('max', '190.0')),
            (('dimension', 'Height'), ('unit', 'in'), ('min', '20.7'), ('max', '30.7')),
            (('dimension', 'Height'), ('unit', 'cm'), ('min', '209.0'), ('max', '209.0')),
            (('dimension', 'Width'), ('unit', 'in'), ('min', '16.9'), ('max', '16.9')),
        }

        assert (
            Controller.aggregate_into_min_max_intervals(field_values, ['dimension', 'unit'], 'value') == expected_result
        )

    @staticmethod
    def test_aggregate_prices() -> None:
        """Test that aggregating prices by currency works as expected"""
        field_values = [
            ((('amount', 1796.280029296875), ('currency', 'AUD'), ('type', 'exact')),),
            (
                (('amount', 1796.280029296875), ('currency', 'AUD'), ('type', 'min')),
                (('amount', 1975.9100341796875), ('currency', 'AUD'), ('type', 'max')),
            ),
            (),
            ((('amount', 140), ('currency', 'EUR'), ('type', 'exact')),),
            (
                (('amount', 123), ('currency', 'EUR'), ('type', 'min')),
                (('amount', 1975.9100341796875), ('currency', 'AUD'), ('type', 'max')),
            ),
        ]

        expected_result = {
            (('currency', 'EUR'), ('min', '123.0'), ('max', '140.0')),
            (('currency', 'AUD'), ('min', '1796.280029296875'), ('max', '1975.9100341796875')),
        }

        assert Controller.aggregate_into_min_max_intervals(field_values, ['currency'], 'amount') == expected_result

    @staticmethod
    def test_aggregate_energy_efficiency() -> None:
        """Test that aggregating energy_efficiency by qualitative and standard_label works as expected"""
        field_values = [
            None,
            (
                (
                    ('exact_percentage', None),
                    ('max_percentage', None),
                    ('min_percentage', None),
                    ('qualitative', 'high'),
                    ('standard_label', None),
                ),
            ),
            (
                (
                    ('exact_percentage', 40.0),
                    ('max_percentage', None),
                    ('min_percentage', None),
                    ('qualitative', None),
                    ('standard_label', None),
                ),
            ),
        ]

        expected_result = {
            (('qualitative', None), ('standard_label', None), ('min', '40.0'), ('max', '40.0')),
            (('qualitative', 'high'), ('standard_label', None), ('min', '-1.0'), ('max', '-1.0')),
        }

        assert Controller.aggregate_energy_efficiency(field_values) == expected_result  # type: ignore

    @staticmethod
    def test_aggregate_production_capacity() -> None:
        """Test that aggregating production_capacity by time_frame and unit works as expected"""
        field_values = [
            ((('quantity', 400000000), ('time_frame', 'Year'), ('type', 'exact'), ('unit', 'Units')),),
            ((('quantity', 60000), ('time_frame', 'Month'), ('type', 'exact'), ('unit', 'Units')),),
            ((('quantity', 1000), ('time_frame', 'Day'), ('type', 'exact'), ('unit', 'Kilograms')),),
            (
                (('quantity', 60), ('time_frame', 'Year'), ('type', 'min'), ('unit', 'Tons')),
                (('quantity', 70), ('time_frame', 'Year'), ('type', 'max'), ('unit', 'Tons')),
            ),
        ]

        expected_result = {
            (('time_frame', 'Month'), ('unit', 'Units'), ('min', '60000.0'), ('max', '60000.0')),
            (('time_frame', 'Day'), ('unit', 'Kilograms'), ('min', '1000.0'), ('max', '1000.0')),
            (('time_frame', 'Year'), ('unit', 'Tons'), ('min', '60.0'), ('max', '70.0')),
            (('time_frame', 'Year'), ('unit', 'Units'), ('min', '400000000.0'), ('max', '400000000.0')),
        }

        assert (
            Controller.aggregate_into_min_max_intervals(field_values, ['time_frame', 'unit'], 'quantity')
            == expected_result
        )

    @staticmethod
    def prepare_data_before_deduplication() -> tuple[
        dict, Literal['CAS: 137-26-8'], list[int], dict, pd.DataFrame, dict
    ]:
        """Helper method for preparing data before deduplication"""
        deduplicated_product = {}
        product_identifier = 'CAS: 137-26-8'
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        df = pd.DataFrame.from_dict(constants.SAMPLE_PRODUCTS, orient='index')
        products_id = Controller.filter_products_by_product_id(
            constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING,  # type: ignore
            product_identifier,
        )
        frequencies: dict[str, dict[str, int]] = defaultdict(dict)

        return deduplicated_product, product_identifier, products_id, df_as_dict, df, frequencies

    @staticmethod
    def test_merge_urls() -> None:
        """Test that all urls are stacked together in 'details' field"""
        # prepare data before deduplication
        deduplicated_product, _, products_id, df_as_dict, _, _ = (
            TestDeduplicateMethods.prepare_data_before_deduplication()
        )

        # provide different root_domain such that it needs to adapt its url to match it
        new_root_domain = 'advancedpressuresystems.ca'
        field = COLUMNS.PAGE_URL.value
        Controller.merge_url(
            df_as_dict,
            products_id,
            field,
            new_root_domain,
            deduplicated_product,
        )

        assert deduplicated_product.get(COLUMNS.DETAILS.value, {}).get(COLUMNS.PAGE_URL.value) == {
            'https://advancedpressuresystems.ca/1',
            'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html',
        }

    @staticmethod
    def test_right_url_value_is_chosen() -> None:
        """Test that the correct value is chosen for page_url"""
        # prepare data before deduplication
        deduplicated_product, _, products_id, df_as_dict, _, _ = (
            TestDeduplicateMethods.prepare_data_before_deduplication()
        )

        # provide different root_domain such that it needs to adapt its url to match it.
        # its previous root domain was harebueng.co.za
        new_root_domain = 'advancedpressuresystems.ca'
        field = COLUMNS.PAGE_URL.value
        Controller.merge_url(
            df_as_dict,
            products_id,
            field,
            new_root_domain,
            deduplicated_product,
        )

        assert deduplicated_product.get(COLUMNS.PAGE_URL.value) == 'https://advancedpressuresystems.ca/1'

    @staticmethod
    def test_lengthiest_value_is_chosen() -> None:
        """Test that the lengthiest value is chosen for every field of MERGE_BY_LENGTHIEST_VALUE"""
        # prepare data before deduplication
        deduplicated_product, _, products_id, df_as_dict, _, _ = (
            TestDeduplicateMethods.prepare_data_before_deduplication()
        )

        for field in MERGE_BY_LENGTHIEST_VALUE:
            Controller.merge_by_the_lengthiest_value(
                df_as_dict,  # type: ignore
                products_id,
                field,
                deduplicated_product,
            )

        field_length_mapping = {k: len(v) for k, v in deduplicated_product.items()}
        assert field_length_mapping == {'description': 525, 'product_summary': 1480}

    @staticmethod
    def test_minimum_value_is_chosen() -> None:
        """Test that the minimum value is chosen for every field of MERGE_BY_MIN_VALUE"""
        # prepare data before deduplication
        deduplicated_product, _, products_id, df_as_dict, _, _ = (
            TestDeduplicateMethods.prepare_data_before_deduplication()
        )

        for field in MERGE_BY_MIN_VALUE:
            Controller.merge_by_the_minimum_value(
                df_as_dict,
                products_id,
                field,
                deduplicated_product,
            )

        assert deduplicated_product == {'id': 9971}

    @staticmethod
    def test_compute_values_to_url_mapping_for_least_frequent_values() -> None:
        """Test that values_to_url_mapping is created for every field of MERGE_BY_LEAST_FREQUENT"""
        deduplicated_product, _, products_id, df_as_dict, df, frequencies = (
            TestDeduplicateMethods.prepare_data_before_deduplication()
        )

        for field in MERGE_BY_LEAST_FREQUENT:
            frequencies[field] = Controller.compute_frequency(df, field)
            field_frequencies = frequencies.get(field, {})

            Controller.merge_by_the_least_frequent_value(
                df_as_dict,
                products_id,
                field,
                field_frequencies,
                deduplicated_product,
            )

        # not needed at this step
        deduplicated_product[COLUMNS.DETAILS.value].pop(COLUMNS.PAGE_URL.value, None)

        assert deduplicated_product[COLUMNS.DETAILS.value] == {
            'product_title': {
                'Rubber Accelerator TMTD IPPD': {'https://advancedpressuresystems.ca/1'},
                'Rubber Accelerator TMQ': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
            },
            'product_name': {
                'Rubber Accelerator': {'https://advancedpressuresystems.ca/1'},
                'TMTD': {'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'},
            },
        }

    @staticmethod
    def test_least_frequent_values_are_chosen() -> None:
        """Test that the least frequent values are chosen for every field of MERGE_BY_LEAST_FREQUENT"""
        # prepare data before deduplication
        deduplicated_product, _, products_id, df_as_dict, df, frequencies = (
            TestDeduplicateMethods.prepare_data_before_deduplication()
        )

        for field in MERGE_BY_LEAST_FREQUENT:
            frequencies[field] = Controller.compute_frequency(df, field)
            field_frequencies = frequencies.get(field, {})

            Controller.merge_by_the_least_frequent_value(
                df_as_dict,
                products_id,
                field,
                field_frequencies,
                deduplicated_product,
                False,
            )

        assert deduplicated_product == {
            'product_title': 'Rubber Accelerator TMTD IPPD',
            'product_name': 'TMTD',
        }

    @staticmethod
    def test_compute_values_to_url_mapping_for_most_frequent_values() -> None:
        """Test that values_to_url_mapping is created well for every field of MERGE_BY_MOST_FREQUENT"""
        # prepare data before deduplication
        deduplicated_product, _, products_id, df_as_dict, df, frequencies = (
            TestDeduplicateMethods.prepare_data_before_deduplication()
        )

        for field in MERGE_BY_MOST_FREQUENT:
            frequencies[field] = Controller.compute_frequency(df, field)
            field_frequencies = frequencies.get(field, {})

            Controller.merge_by_the_most_frequent_value(
                df_as_dict,
                products_id,
                field,
                field_frequencies,
                deduplicated_product,
            )

        # not needed at this step
        deduplicated_product[COLUMNS.DETAILS.value].pop(COLUMNS.PAGE_URL.value, None)

        assert deduplicated_product[COLUMNS.DETAILS.value] == {
            'unspsc': {
                'Curing agents': {'https://advancedpressuresystems.ca/1'},
                'Faucets or taps': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
            },
            'root_domain': {
                'advancedpressuresystems.ca': {'https://advancedpressuresystems.ca/1'},
                'harebueng.co.za': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
            },
            'brand': {
                'Nutrena': {'https://advancedpressuresystems.ca/1'},
            },
        }

    @staticmethod
    def test_most_frequent_values_are_chosen() -> None:
        """Test that the most frequent values are chosen for every field of MERGE_BY_MOST_FREQUENT"""
        deduplicated_product, _, products_id, df_as_dict, df, frequencies = (
            TestDeduplicateMethods.prepare_data_before_deduplication()
        )

        for field in MERGE_BY_MOST_FREQUENT:
            frequencies[field] = Controller.compute_frequency(df, field)
            field_frequencies = frequencies.get(field, {})

            Controller.merge_by_the_most_frequent_value(
                df_as_dict,
                products_id,
                field,
                field_frequencies,
                deduplicated_product,
                False,
            )

        expected_result = {
            'unspsc': 'Curing agents',
            'root_domain': 'harebueng.co.za',
            'brand': 'Nutrena',
        }

        assert deduplicated_product == expected_result

    @staticmethod
    def test_frequency_is_updated_after_deduplication() -> None:
        """Test that frequencies are updated after merging"""
        deduplicated_product, _, products_id, df_as_dict, df, frequencies = (
            TestDeduplicateMethods.prepare_data_before_deduplication()
        )

        for field in MERGE_BY_MOST_FREQUENT:
            frequencies[field] = Controller.compute_frequency(df, field)
            field_frequencies = frequencies.get(field, {})

            Controller.merge_by_the_most_frequent_value(
                df_as_dict,
                products_id,
                field,
                field_frequencies,
                deduplicated_product,
                False,
            )

        frequencies_as_dict = {k: dict(v) for k, v in frequencies.items()}
        expected_result = {
            'unspsc': {'Pipe connectors': 1, 'Curing agents': 2},
            'root_domain': {'studio-atcoat.com': 1, 'harebueng.co.za': 2},
            'brand': {'DeRoyal': 1, 'Nutrena': 2},
        }

        assert frequencies_as_dict == expected_result

    @staticmethod
    def test_deduplication_update() -> None:
        """
        Test that after deduplication, products with the same product_identifier
        are removed and replaced with a single deduplicated product.
        """
        _, _, products_id, df_as_dict, _, _ = TestDeduplicateMethods.prepare_data_before_deduplication()
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        initial_products = list(df_as_dict.keys())
        merge_group(df_as_dict, products_id, frequencies)
        deduplicated_products = list(df_as_dict.keys())
        assert len(initial_products) == len(deduplicated_products) + 1
        assert 10275 not in deduplicated_products


class TestStandardizationMethods:
    """Test methods used for standardization, used for converting dataframe back to parquet"""

    @staticmethod
    def test_merge_price_intervals():
        """Test that merging price intervals works as expected"""
        initial_rows = [
            [
                {'currency': 'USD', 'min': '26.989999771118164', 'max': '26.989999771118164'},
                {'currency': 'USD', 'min': '44.9900016784668', 'max': '44.9900016784668'},
            ],
            [
                {'currency': 'USD', 'min': '26.989999771118164', 'max': '26.989999771118164'},
            ],
            [
                {'currency': 'USD', 'min': '26.989999771118164', 'max': '26.989999771118164'},
                {'currency': 'EUR', 'min': '44.9900016784668', 'max': '54.9900016784668'},
            ],
        ]

        expected_result = [
            [{'min': '26.989999771118164', 'max': '44.9900016784668', 'currency': 'USD'}],
            [{'min': '26.989999771118164', 'max': '26.989999771118164', 'currency': 'USD'}],
            [
                {'min': '26.989999771118164', 'max': '26.989999771118164', 'currency': 'USD'},
                {'min': '44.9900016784668', 'max': '54.9900016784668', 'currency': 'EUR'},
            ],
        ]
        results = [StandardizeController.standardize_price(row) for row in initial_rows]

        assert results == expected_result

    @staticmethod
    def test_merge_production_capacity_intervals():
        """Test that merging production_capacity intervals works as expected"""
        initial_rows = [
            [{'min': '5000000.0', 'max': '5000000.0', 'unit': 'Units', 'time_frame': 'Month'}],
            [
                {'time_frame': 'Year', 'unit': 'Tons', 'min': '60', 'max': '60'},
                {'time_frame': 'Year', 'unit': 'Tons', 'min': '70', 'max': '70'},
            ],
        ]

        expected_result = [
            [{'min': '5000000.0', 'max': '5000000.0', 'unit': 'Units', 'time_frame': 'Month'}],
            [{'min': '60.0', 'max': '70.0', 'unit': 'Tons', 'time_frame': 'Year'}],
        ]
        results = [StandardizeController.standardize_production_capacity(row) for row in initial_rows]

        assert results == expected_result

    @staticmethod
    def test_merge_purity_pressure_rating_power_rating_intervals():
        """Test that merging purity_pressure_rating_power_rating intervals works as expected"""
        initial_rows = [
            [{'qualitative': False, 'unit': 'W', 'min': '95.0', 'max': '95.0'}],
            [
                {'qualitative': False, 'unit': 'kW', 'min': '0.37', 'max': '0.37'},
                {'qualitative': False, 'unit': 'kW', 'min': '2.2', 'max': '2.2'},
            ],
            [
                {'qualitative': False, 'unit': 'kW', 'min': '15.0', 'max': '15.0'},
                {'qualitative': False, 'unit': 'Mhz', 'min': '22.0', 'max': '22.0'},
            ],
        ]

        expected_result = [
            [{'qualitative': False, 'unit': 'W', 'min': '95.0', 'max': '95.0'}],
            [
                {'qualitative': False, 'unit': 'kW', 'min': '0.37', 'max': '2.2'},
            ],
            [
                {'qualitative': False, 'unit': 'kW', 'min': '15.0', 'max': '15.0'},
                {'qualitative': False, 'unit': 'Mhz', 'min': '22.0', 'max': '22.0'},
            ],
        ]
        results = [StandardizeController.standardize_purity_pressure_rating_power_rating(row) for row in initial_rows]

        assert results == expected_result

    @staticmethod
    def test_merge_size_intervals():
        """Test that merging size intervals works as expected"""
        initial_rows = [
            [
                {'dimension': 'Length', 'unit': 'in', 'min': '3.5', 'max': '3.5'},
                {'dimension': 'Width', 'unit': 'in', 'min': '2.25', 'max': '2.25'},
            ],
            [{'dimension': 'Volume', 'unit': 'gal', 'min': '1.0', 'max': '1.0'}],
            [
                {'dimension': 'Length', 'unit': 'in', 'min': '3.5', 'max': '3.5'},
                {'dimension': 'Length', 'unit': 'inch', 'min': '12.25', 'max': '12.25'},
                {'dimension': 'Length', 'unit': 'in', 'min': '5.25', 'max': '7.25'},
            ],
        ]

        expected_result = [
            [
                {'dimension': 'Length', 'unit': 'in', 'min': '3.5', 'max': '3.5'},
                {'dimension': 'Width', 'unit': 'in', 'min': '2.25', 'max': '2.25'},
            ],
            [{'dimension': 'Volume', 'unit': 'gal', 'min': '1.0', 'max': '1.0'}],
            [
                {'dimension': 'Length', 'unit': 'in', 'min': '3.5', 'max': '7.25'},
                {'dimension': 'Length', 'unit': 'inch', 'min': '12.25', 'max': '12.25'},
            ],
        ]
        results = [StandardizeController.standardize_size(row) for row in initial_rows]

        assert results == expected_result
