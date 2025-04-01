from copy import deepcopy
from collections import defaultdict
import pandas as pd


from controller import Controller
from main import merge_by_product_identifier
from settings import (
    OUTPUT_TEST_FILE,
    MERGE_BY_COMPLETING,
    MERGE_BY_LENGTHIEST_VALUE,
    MERGE_BY_MIN_VALUE,
    MERGE_BY_MOST_FREQUENT,
    MERGE_BY_LEAST_FREQUENT,
    OUTPUT_E2E_FILE,
    FILE_PATH,
    COLUMNS,
)
from tests import constants


class TestIntegration:
    """Integration tests"""

    deduplicated_product_id = 9971
    len_deduplicated_products = 21881

    @staticmethod
    def test_e2e() -> None:
        """Test the flow successfully writes the resulted file"""
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
        df = Controller.convert_to_dataframe(df_as_dict)
        df.to_parquet(OUTPUT_E2E_FILE)

        assert len(df) == TestIntegration.len_deduplicated_products

    @staticmethod
    def test_convert_dict_back_to_parquet() -> None:
        """Test that converting dictionary back to parquet is successful"""
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING
        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        df = Controller.convert_to_dataframe(df_as_dict)
        df.to_parquet(OUTPUT_TEST_FILE)
        df_as_dict = df.to_dict(orient='index')
        assert len(df_as_dict.keys()) == 3

    @staticmethod
    def test_details_for_merge_by_lengthiest_value_were_modified_correctly() -> None:
        """
        Test that every field from 'details' corresponding to MERGE_BY_LENGTHIEST_VALUE was aggregated correctly
        for product with id 9971. They should be empty because we dont store them in 'details'
        """
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        details = df_as_dict.get(TestIntegration.deduplicated_product_id, {}).get(COLUMNS.DETAILS.value, {})
        deduplicated_fields = {k: len(v) for k, v in details.items() if k in MERGE_BY_LENGTHIEST_VALUE}

        assert {} == deduplicated_fields

    @staticmethod
    def test_details_for_merge_by_least_frequent_were_modified_correctly() -> None:
        """
        Test that every field from 'details' corresponding to MERGE_BY_LEAST_FREQUENT was aggregated correctly
        for product with id 9971
        """
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        details = df_as_dict.get(TestIntegration.deduplicated_product_id, {}).get(COLUMNS.DETAILS.value, {})
        deduplicated_fields = {k: v for k, v in details.items() if k in MERGE_BY_LEAST_FREQUENT}

        expected_result = {
            'product_title': {
                'Rubber Accelerator TMQ': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                'Rubber Accelerator TMTD IPPD': {'https://advancedpressuresystems.ca/1'},
            },
            'product_name': {
                'TMTD': {'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'},
                'Rubber Accelerator': {'https://advancedpressuresystems.ca/1'},
            },
        }

        assert expected_result == deduplicated_fields

    @staticmethod
    def test_details_for_merge_by_most_frequent_were_modified_correctly() -> None:
        """
        Test that every field from 'details' corresponding to MERGE_BY_MOST_FREQUENT was aggregated correctly
        for product with id 9971
        """
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        details = df_as_dict.get(TestIntegration.deduplicated_product_id, {}).get(COLUMNS.DETAILS.value, {})
        deduplicated_fields = {k: v for k, v in details.items() if k in MERGE_BY_MOST_FREQUENT}

        expected_result = {
            'brand': {
                'Nutrena': {'https://advancedpressuresystems.ca/1'},
            },
            'unspsc': {
                'Faucets or taps': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                'Curing agents': {'https://advancedpressuresystems.ca/1'},
            },
            'root_domain': {
                'harebueng.co.za': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                'advancedpressuresystems.ca': {'https://advancedpressuresystems.ca/1'},
            },
        }

        assert expected_result == deduplicated_fields

    @staticmethod
    def test_details_for_merge_by_completing_were_modified_correctly() -> None:
        """
        Test that every field from 'details' corresponding to MERGE_BY_COMPLETING was aggregated correctly
        for product with id 9971
        """
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        details = df_as_dict.get(TestIntegration.deduplicated_product_id, {}).get(COLUMNS.DETAILS.value, {})
        deduplicated_fields = {k: v for k, v in details.items() if k in MERGE_BY_COMPLETING}

        expected_result = {
            'intended_industries': {
                'Rubber Processing': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                'Agriculture': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                'Rubber Industry': {'https://advancedpressuresystems.ca/1'},
            },
            'applicability': {
                'Rubber Processing': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html',
                    'https://advancedpressuresystems.ca/1',
                },
                'Fungicide': {'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'},
                'Seed Soaking': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
            },
            'eco_friendly': {
                'True': {'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'},
            },
            'ethical_and_sustainability_practices': {
                'adhering to environmental standards and regulations': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                'designed to be recyclable': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html',
                    'https://advancedpressuresystems.ca/1',
                },
                'meet the E1 formaldehyde standards': {'https://advancedpressuresystems.ca/1'},
                'CVI Green products': {'https://advancedpressuresystems.ca/1'},
                'extremely low in emissions': {'https://advancedpressuresystems.ca/1'},
                'ISO 9001:2015 standard': {'https://advancedpressuresystems.ca/1'},
            },
            'production_capacity': {
                (('quantity', 60), ('time_frame', 'Year'), ('type', 'min'), ('unit', 'Tons')): {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                (('quantity', 70), ('time_frame', 'Year'), ('type', 'max'), ('unit', 'Tons')): {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                (('quantity', 60000), ('time_frame', 'Month'), ('type', 'exact'), ('unit', 'Units')): {
                    'https://advancedpressuresystems.ca/1'
                },
            },
            'price': {
                (('amount', 1.809999942779541), ('currency', 'USD'), ('type', 'min')): {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                (('amount', 1.899999976158142), ('currency', 'USD'), ('type', 'max')): {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
            },
            'materials': {
                'Ceramic': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html',
                    'https://advancedpressuresystems.ca/1',
                },
                'Ceramic powder': {'https://advancedpressuresystems.ca/1'},
            },
            'ingredients': {
                'Vanilla': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html',
                    'https://advancedpressuresystems.ca/1',
                },
                'Salt': {'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'},
                'Spiced': {'https://advancedpressuresystems.ca/1'},
            },
            'manufacturing_countries': {'PK': {'https://advancedpressuresystems.ca/1'}},
            'manufacturing_year': {
                '-1': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html',
                    'https://advancedpressuresystems.ca/1',
                }
            },
            'manufacturing_type': {
                'Turnkey': {'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'}
            },
            'customization': {
                'Various types of grinding wheels available': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                'Various colors available': {'https://advancedpressuresystems.ca/1'},
            },
            'packaging_type': {'Cartons': {'https://advancedpressuresystems.ca/1'}},
            'form': {
                'Powder': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html',
                    'https://advancedpressuresystems.ca/1',
                },
                'Granules': {'https://advancedpressuresystems.ca/1'},
            },
            'size': {
                (
                    ('dimension', 'Diameter'),
                    ('qualitative', False),
                    ('type', 'min'),
                    ('unit', 'mm'),
                    ('value', '115'),
                ): {'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'},
                (
                    ('dimension', 'Diameter'),
                    ('qualitative', False),
                    ('type', 'max'),
                    ('unit', 'mm'),
                    ('value', '450'),
                ): {'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'},
                (
                    ('dimension', 'Length'),
                    ('qualitative', False),
                    ('type', 'exact'),
                    ('unit', 'mm'),
                    ('value', '127'),
                ): {'https://advancedpressuresystems.ca/1'},
                (('dimension', 'Width'), ('qualitative', False), ('type', 'exact'), ('unit', 'mm'), ('value', '82')): {
                    'https://advancedpressuresystems.ca/1'
                },
            },
            'color': {
                (('original', 'Blue'), ('simple', 'White')): {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                (('original', 'Blue'), ('simple', 'Blue')): {'https://advancedpressuresystems.ca/1'},
            },
            'purity': {
                (('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '96.0')): {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                (('qualitative', False), ('type', 'exact'), ('unit', None), ('value', '97.0')): {
                    'https://advancedpressuresystems.ca/1'
                },
                (('qualitative', True), ('type', 'exact'), ('unit', None), ('value', 'high')): {
                    'https://advancedpressuresystems.ca/1'
                },
            },
            'energy_efficiency': {
                (
                    ('exact_percentage', 40.0),
                    ('max_percentage', None),
                    ('min_percentage', None),
                    ('qualitative', 'high'),
                    ('standard_label', None),
                ): {'https://advancedpressuresystems.ca/1'},
            },
            'pressure_rating': {
                (('qualitative', True), ('type', 'exact'), ('unit', None), ('value', 'high')): {
                    'https://advancedpressuresystems.ca/1'
                }
            },
            'power_rating': {
                (('qualitative', False), ('type', 'exact'), ('unit', 'W'), ('value', '1200.0')): {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                (('qualitative', True), ('type', 'exact'), ('unit', None), ('value', 'high')): {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
            },
            'quality_standards_and_certifications': {
                'ISO Certified': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                'HG/T 2334-2007': {'https://advancedpressuresystems.ca/1'},
            },
            'miscellaneous_features': {
                'Initial Melting Point: At least 142.0°C': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                'Heating Loss Percentage: ≤ 0.40%': {
                    'https://www.harebueng.co.za/antiscorching-pvi-antiscorching-pvi-suppliers-poland.html'
                },
                'Solubility in benzene, acetone, chloroform, CS2, and partially soluble in alcohol, diethyl ether, and CCl4': {
                    'https://advancedpressuresystems.ca/1'
                },
                'Insoluble in water, gasoline, and alkali with lower concentrations': {
                    'https://advancedpressuresystems.ca/1'
                },
            },
        }

        assert expected_result == deduplicated_fields

    @staticmethod
    def test_merge_by_completing_fields_were_modified_correctly() -> None:
        """Test that every field from MERGE_BY_COMPLETING was modified correctly for product with id 9971"""
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        deduplicated_fields = {
            field: df_as_dict.get(TestIntegration.deduplicated_product_id, {}).get(field)
            for field in MERGE_BY_COMPLETING
        }
        expected_result = {
            'intended_industries': {'Agriculture', 'Rubber Industry', 'Rubber Processing'},
            'applicability': {'Seed Soaking', 'Fungicide', 'Rubber Processing'},
            'ethical_and_sustainability_practices': {
                'designed to be recyclable',
                'CVI Green products',
                'ISO 9001:2015 standard',
                'adhering to environmental standards and regulations',
                'extremely low in emissions',
                'meet the E1 formaldehyde standards',
            },
            'materials': {'Ceramic', 'Ceramic powder'},
            'ingredients': {'Spiced', 'Vanilla', 'Salt'},
            'manufacturing_countries': {'PK'},
            'purity': {
                (('qualitative', True), ('unit', None), ('min', 'high'), ('max', 'high')),
                (('qualitative', False), ('unit', None), ('min', '96.0'), ('max', '97.0')),
            },
            'energy_efficiency': {
                (('qualitative', 'high'), ('standard_label', None), ('min', '40.0'), ('max', '40.0'))
            },
            'pressure_rating': {(('qualitative', True), ('unit', None), ('min', 'high'), ('max', 'high'))},
            'eco_friendly': {None, True},
            'power_rating': {
                (('qualitative', False), ('unit', 'W'), ('min', '1200.0'), ('max', '1200.0')),
                (('qualitative', True), ('unit', None), ('min', 'high'), ('max', 'high')),
            },
            'quality_standards_and_certifications': {'HG/T 2334-2007', 'ISO Certified'},
            'form': {'Powder', 'Granules'},
            'manufacturing_year': {-1},
            'production_capacity': {
                (('min', '60.0'), ('time_frame', 'Year'), ('unit', 'Tons'), ('max', '70.0')),
                (('min', '60000.0'), ('time_frame', 'Month'), ('unit', 'Units'), ('max', '60000.0')),
            },
            'price': {(('min', '1.809999942779541'), ('currency', 'USD'), ('max', '1.899999976158142'))},
            'manufacturing_type': {'Turnkey'},
            'customization': {'Various colors available', 'Various types of grinding wheels available'},
            'packaging_type': {'Cartons'},
            'size': {
                (('dimension', 'Width'), ('unit', 'mm'), ('min', '82.0'), ('max', '82.0')),
                (('dimension', 'Diameter'), ('unit', 'mm'), ('min', '115.0'), ('max', '450.0')),
                (('dimension', 'Length'), ('unit', 'mm'), ('min', '127.0'), ('max', '127.0')),
            },
            'color': {(('original', 'Blue'), ('simple', 'Blue, White'))},
            'miscellaneous_features': {
                'Initial Melting Point: At least 142.0°C',
                'Heating Loss Percentage: ≤ 0.40%',
                'Insoluble in water, gasoline, and alkali with lower concentrations',
                'Solubility in benzene, acetone, chloroform, CS2, and partially soluble in alcohol, diethyl ether, and CCl4',
            },
        }

        assert deduplicated_fields == expected_result

    @staticmethod
    def test_product_identifier_was_modified_correctly() -> None:
        """Test that product product_identifier is modified correctly for product with id 9971"""
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        product_identifier = df_as_dict.get(TestIntegration.deduplicated_product_id, {}).get(
            COLUMNS.PRODUCT_IDENTIFIER.value, {}
        )
        expected_result = 'CAS: 137-26-8'

        assert product_identifier == expected_result

    @staticmethod
    def test_merge_by_lengthiest_value_fields_were_modified_correctly() -> None:
        """Test that every field from MERGE_BY_LENGTHIEST_VALUE was modified correctly for product with id 9971"""
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        deduplicated_fields = [
            len(df_as_dict.get(TestIntegration.deduplicated_product_id, {}).get(field))  # type: ignore
            for field in MERGE_BY_LENGTHIEST_VALUE
        ]
        expected_result = [525, 1480]

        assert deduplicated_fields == expected_result

    @staticmethod
    def test_id_is_modified_correctly() -> None:
        """Test that every field from MERGE_BY_MIN_VALUE (only id) was modified correctly for product with id 9971"""
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        deduplicated_fields = [
            df_as_dict.get(TestIntegration.deduplicated_product_id, {}).get(field) for field in MERGE_BY_MIN_VALUE
        ]
        expected_result = [9971]

        assert deduplicated_fields == expected_result

    @staticmethod
    def test_merge_by_least_frequent_fields_were_modified_correctly():
        """Test that every field from MERGE_BY_LEAST_FREQUENT was modified correctly for product with id 9971"""
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        deduplicated_fields = [
            df_as_dict.get(TestIntegration.deduplicated_product_id, {}).get(field) for field in MERGE_BY_LEAST_FREQUENT
        ]
        expected_result = ['TMTD', 'Rubber Accelerator TMTD IPPD']

        assert deduplicated_fields == expected_result

    @staticmethod
    def test_merge_by_most_frequent_fields_were_modified_correctly() -> None:
        """Test that every field from MERGE_BY_MOST_FREQUENT was modified correctly for product with id 9971"""
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        deduplicated_fields = [
            df_as_dict.get(TestIntegration.deduplicated_product_id, {}).get(field) for field in MERGE_BY_MOST_FREQUENT
        ]
        expected_result = ['Curing agents', 'harebueng.co.za', 'Nutrena']

        assert deduplicated_fields == expected_result

    @staticmethod
    def test_products_remain_the_same() -> None:
        """
        Test that the products that should not be deduplicated (do not have other instances of themselves)
        remain the same.
        Apart from product with id 9971, the rest should be the same, exception being 10275, which was merged into 9971
        """
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        before = deepcopy(df_as_dict)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        final_products = list(df_as_dict.keys())
        modified = [before[product] != df_as_dict[product] for product in final_products]

        assert modified == [False, False, True]

    @staticmethod
    def test_remove_correct_product() -> None:
        """Test that product with id 10275 is merged and is not existent anymore"""
        df_as_dict = deepcopy(constants.SAMPLE_PRODUCTS)
        before = deepcopy(df_as_dict)
        frequencies = deepcopy(constants.COMPUTE_FREQUENCIES_RESULT)
        product_identifier_to_pid = constants.SAMPLE_PRODUCTS_ID_TO_PRODUCT_IDENTIFIER_MAPPING

        merge_by_product_identifier(df_as_dict, product_identifier_to_pid, frequencies)  # type: ignore

        initial_products = list(before.keys())
        deduplicated_products = list(df_as_dict.keys())

        assert initial_products == deduplicated_products + [10275]
