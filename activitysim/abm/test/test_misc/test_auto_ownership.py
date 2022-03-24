import logging
import pytest
import os
import pandas as pd

# import models is necessary to initalize the model steps with orca
from activitysim.abm import models
from activitysim.core import pipeline, config
from activitysim.core import tracing

logger = logging.getLogger(__name__)

# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope='module')
def module(prepare_module_inputs) -> str:
    """
    A pytest fixture that returns the data folder location.
    :return: folder location for any necessary data to initialize the tests
    """
    return 'auto_ownership'


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope='module')
def tables() -> dict[str, str]:
    """
    A pytest fixture that returns the "mock" tables to build pipeline dataframes. The
    key-value pair is the name of the table and the index column.
    :return: dict
    """
    return {
        'land_use': 'zone_id',
        'persons': 'person_id',
        'households': 'household_id',
        'accessibility': 'zone_id'
    }


# Used by conftest.py initialize_pipeline method
# Set to true if you need to read skims into the pipeline
@pytest.fixture(scope='module')
def initialize_network_los() -> bool:
    """
    A pytest boolean fixture indicating whether network skims should be read from the
    fixtures test data folder.
    :return: bool
    """
    return False


def test_auto_ownership(initialize_pipeline: pipeline.Pipeline, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    # run model step
    pipeline.run(models=['auto_ownership_simulate'])
    
    # get the updated pipeline data
    household_df = pipeline.get_table('households')

    tracing.print_summary('auto_ownership', household_df.auto_ownership, describe=True)

    # compare with targets
    # TODO check the value of auto_ownership
    if validate_model_against_target(prepare_targets, household_df):
        logger.info("Model result matches target")
    else:
        logger.info("Model result does not match target")

# fetch/prepare existing files for model inputs
# e.g. read accessibilities.csv from ctramp result, rename columns, write out to accessibility.csv which is the input to activitysim
@pytest.fixture(scope='module')
def prepare_module_inputs() -> None:
    """
    Prepare accessibility file from ctramp run into input file activitysim expects
    1. renaming columns
    2. write out table

    Prepare household, person, landuse data into format activitysim expects
    1. renaming columns
    2. write out table

    :return: None
    """

    # this is the ctramp accessibility output file
    # can be downloaded from 
    # https://wsponlinenam.sharepoint.com/:x:/r/sites/US-TM2ConversionProject/Shared%20Documents/Task%203%20ActivitySim/model_results/
    # ver12_new_inputs/ctramp_output/accessibilities.csv?d=wef3dbcd2186c42518a9e9558a15a2ca3&csf=1&web=1&e=6whUv0
    input_accessibility_file = os.path.join('test', 'auto_ownership', 'data', 'accessibilities.csv')

    accessibility_df = pd.read_csv(
        input_accessibility_file
    )

    # rename columns
    # this dictionary is developed based on mtc ctramp code
    # https://github.com/BayAreaMetro/travel-model-two/blob/27e3ad6e5a6c71120e3e513fe423ecaac372c63a/core/src/java/com/pb/mtctm2/abm/accessibilities/AccessibilitiesTable.java#L24-L61
    # TODO confirm this is the corrent column names
    accessibility_columns_dict = {
        'column_1': 'nonmandatory_auto_accessibility',
        'column_2': 'nonmandatory_transit_accessibility',
        'column_3': 'nonmandatory_nm_accessibility',
        'column_4': 'nonmandatory_sov0_accessibility',
        'column_5': 'nonmandatory_sov1_accessibility',
        'column_6': 'nonmandatory_sov2_accessibility',
        'column_7': 'nonmandatory_hov0_accessibility',
        'column_8': 'nonmandatory_hov1_accessibility',
        'column_9': 'nonmandatory_hov2_accessibility',
        'column_10': 'shop_hov_insufficient_accessibility',
        'column_11': 'shop_hov_sufficient_accessibility',
        'column_12': 'shop_hov_oversufficient_accessibility',
        'column_13': 'maint_hov_insufficient_accessibility',
        'column_14': 'maint_hov_sufficient_accessibility',
        'column_15': 'maint_hov_oversufficient_accessibility',
        'column_16': 'eat_hov_insufficient_accessibility',
        'column_17': 'eat_hov_sufficient_accessibility',
        'column_18': 'eat_hov_oversufficient_accessibility',
        'column_19': 'visit_hov_insufficient_accessibility',
        'column_20': 'visit_hov_sufficient_accessibility',
        'column_21': 'visit_hov_oversufficient_accessibility',
        'column_22': 'discr_hov_insufficient_accessibility',
        'column_23': 'discr_hov_sufficient_accessibility',
        'column_24': 'discr_hov_oversufficient_accessibility',
        'column_25': 'escort_hov_insufficient_accessibility',
        'column_26': 'escort_hov_sufficient_accessibility',
        'column_27': 'escort_hov_oversufficient_accessibility',
        'column_28': 'shop_sov_insufficient_accessibility',
        'column_29': 'shop_sov_sufficient_accessibility',
        'column_30': 'shop_sov_oversufficient_accessibility',
        'column_31': 'maint_sov_insufficient_accessibility',
        'column_32': 'maint_sov_sufficient_accessibility',
        'column_33': 'maint_sov_oversufficient_accessibility',
        'column_40': 'discr_sov_insufficient_accessibility',
        'column_41': 'discr_sov_sufficient_accessibility',
        'column_42': 'discr_sov_oversufficient_accessibility',
        'column_45': 'total_emp_accessibility',
        'column_47': 'hh_walktransit_accessibility',
        'mgra' : 'zone_id'
    }

    accessibility_df.rename(columns=accessibility_columns_dict, inplace = True)

    accessibility_df.to_csv(
        os.path.join('test', 'auto_ownership', 'data', 'accessibility.csv'),
        index = False
    )

    ## TODO
    # annotate household, person, and land_use data

# TODO 
# create target database from existing run
@pytest.fixture(scope='module')
def prepare_targets() -> pd.DataFrame:
    """
    Prepare auto ownership target data from existing ctramp run

    :return: pd.DataFrame
    """

    # ctramp run result
    # https://wsponlinenam.sharepoint.com/:x:/r/sites/US-TM2ConversionProject/Shared%20Documents/Task%203%20ActivitySim/model_results/
    # ver12_new_inputs/ctramp_output/aoResults.csv?d=wba2db5a7cd8841ae822cc4234038b258&csf=1&web=1&e=KmHeWN

# TODO
# flesh out assert function
def validate_model_against_target(target_df: pd.DataFrame, model_df: pd.DataFrame) -> bool:
    """
    assert funtion that compares model summary with target
    e.g. loop through each cell in the summary table, if % diff within threshold then model matches target

    :return: bool
    """

    return True