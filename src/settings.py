from enum import Enum


OUTPUT_FILE = 'dataset/deduplicated.parquet'
OUTPUT_E2E_FILE = 'dataset/deduplicated_e2e_test.parquet'
OUTPUT_TEST_FILE = 'dataset/deduplicated_test.parquet'
FILE_PATH = 'dataset/db.snappy.parquet'
SAMPLE_FILE_PATH = 'dataset/sample.parquet'


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
    ID = 'id'  # added by myself
    DETAILS = 'details'  # added by myself


MERGE_BY_MOST_FREQUENT = [COLUMNS.UNSPSC.value, COLUMNS.ROOT_DOMAIN.value, COLUMNS.BRAND.value]
MERGE_BY_LEAST_FREQUENT = [COLUMNS.PRODUCT_NAME.value, COLUMNS.PRODUCT_TITLE.value]
MERGE_BY_MIN_VALUE = [COLUMNS.ID.value]
MERGE_BY_LENGTHIEST_VALUE = [COLUMNS.DESCRIPTION.value, COLUMNS.PRODUCT_SUMMARY.value]
MERGE_BY_COMPLETING = [
    COLUMNS.INTENDED_INDUSTRIES.value,
    COLUMNS.APPLICABILITY.value,
    COLUMNS.ETHICAL_AND_SUSTAINABILITY_PRACTICES.value,
    COLUMNS.MATERIALS.value,
    COLUMNS.INGREDIENTS.value,
    COLUMNS.MANUFACTURING_COUNTRIES.value,
    COLUMNS.PURITY.value,
    COLUMNS.ENERGY_EFFICIENCY.value,
    COLUMNS.PRESSURE_RATING.value,
    COLUMNS.ECO_FRIENDLY.value,
    COLUMNS.POWER_RATING.value,
    COLUMNS.QUALITY_STANDARDS_AND_CERTIFICATIONS.value,
    COLUMNS.FORM.value,
    COLUMNS.MANUFACTURING_YEAR.value,
    COLUMNS.PRODUCTION_CAPACITY.value,
    COLUMNS.PRICE.value,
    COLUMNS.MANUFACTURING_TYPE.value,
    COLUMNS.CUSTOMIZATION.value,
    COLUMNS.PACKAGING_TYPE.value,
    COLUMNS.SIZE.value,
    COLUMNS.COLOR.value,
    COLUMNS.MISCELLANEOUS_FEATURES.value,
]

LIST_OF_DICT = [
    COLUMNS.PRODUCTION_CAPACITY.value,
    COLUMNS.PRICE.value,
    COLUMNS.SIZE.value,
    COLUMNS.COLOR.value,
    COLUMNS.PURITY.value,
    COLUMNS.PRESSURE_RATING.value,
    COLUMNS.POWER_RATING.value,
]
