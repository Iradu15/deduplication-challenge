from enum import Enum


OUTPUT_FILE = 'dataset/deduplicated.parquet'
OUTPUT_E2E_FILE = 'dataset/deduplicated_e2e_test.parquet'
OUTPUT_TEST_FILE = 'dataset/deduplicated_test.parquet'
FILE_PATH = 'db.snappy.parquet'
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
    # BASE_ROOT_DOMAIN = 'base_root_domain'  # added by myself, needed to differentiate between domain.eu / domain.com
    DETAILS = 'details'  # added by myself, needed to differentiate between domain.eu / domain.com


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


duplicate_product_identifiers = (
    'Part_Number: F1200B-V',
    'UPC: 39026',
    'SKU: CMP-CNVRT-HDMI2VGA-WHT-KIT',
    'SKU: 1783FXDMIDL',
    'Part_Number: F600BT-V,Part_Number: F600B-V,Part_Number: F600B4F-V,Part_Number: F1200B-V',
    'CAS: 793-24-8',
    'SKU: 1783SUDCOMK',
    'SKU: 113129',
    'Part_Number: IHB21K6',
    'Part_Number: 40350',
    'SKU: 1783TSYBLD',
    'Part_Number: 56000',
    'UPC: 123456789',
    'Part_Number: ZKAD-C3M',
    'Part_Number: 58773',
    'Part_Number: 70154BKXS',
    'Part_Number: 53007',
    'Part_Number: F1600B',
    'SKU: 109139',
    'Part_Number: 52170',
    'Part_Number: REU-01075',
    'Part_Number: 52943',
    'SKU: WP1540AW',
    'Part_Number: 40149',
    'Product_Code: BLD7714-SM',
    'Part_Number: 66500',
    'Part_Number: 52940',
    'Part_Number: 95848',
    'Part_Number: 90155',
    'Part_Number: 30440',
    'SKU: Not Available',
    'Part_Number: 29790',
    'Part_Number: FJ2150_NAVY',
    'Part_Number: F3200S',
    'Part_Number: 32767',
    'Product_Code: B07H9H6SPN',
    'SKU: 34246',
    'Part_Number: IS410',
    'Part_Number: F1200B,Part_Number: F1200BT,Part_Number: F1200B-V',
    'Product_Code: IHB21K6',
    'Part_Number: 53085',
    'Part_Number: F600S,Part_Number: F600S4,Part_Number: F600S4F,Part_Number: F600SF',
    'Product_Code: HT-B8110',
    'SKU: BNDL-V0000-2X',
    'Part_Number: REU-OADA-C16Z10-B770',
    'Part_Number: 69622',
    'Part_Number: REU-98707-2346',
    'CAS: 137-26-8',
    'Part_Number: 0986435505',
    'Part_Number: HP108',
)
