import logging
import pytest
import os
import shutil
import pandas as pd

# import models is necessary to initalize the model steps with orca
from activitysim.abm import models
from activitysim.core import pipeline, config
from activitysim.core import tracing

logger = logging.getLogger(__name__)

# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope='module')
def module() -> str:
    """
    A pytest fixture that returns the data folder location.
    :return: folder location for any necessary data to initialize the tests
    """
    return 'cdap'


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope='module')
def tables(prepare_module_inputs) -> dict[str, str]:
    """
    A pytest fixture that returns the "mock" tables to build pipeline dataframes. The
    key-value pair is the name of the table and the index column.
    :return: dict
    """
    return {
        'land_use': 'MAZ_ORIGINAL',
        'persons': 'person_id',
        'households': 'household_id',
        'accessibility': 'MAZ_ORIGINAL'
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

# TODO prepare data with info from travel model two result

# make a reconnect_pipeline internal to cdap
@pytest.mark.skipif(os.path.isfile('test/cdap/output/pipeline.h5'), reason = "no need to recreate pipeline store if alreayd exist")
def test_prepare_input_pipeline(initialize_pipeline: pipeline.Pipeline, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    # run model step
    pipeline.run(models=[
        'initialize_landuse',
        'initialize_households'
        ]
    )

    pipeline.close_pipeline()

def test_cdap_from_pipeline(reconnect_pipeline: pipeline.Pipeline, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    # run model step
    pipeline.run(models=[
        'cdap_simulate'
        ],
        resume_after = 'initialize_households'
    )
    
    # get the updated pipeline data
    person_df = pipeline.get_table('persons')

    tracing.print_summary('cdap_simulate', person_df.cdap_activity, describe=True)

    # compare with targets
    # TODO check the value of auto_ownership
    if validate_model_against_target(prepare_targets, person_df):
        logger.info("Model result matches target")
    else:
        logger.info("Model result does not match target")

# fetch/prepare existing files for model inputs
# e.g. read accessibilities.csv from ctramp result, rename columns, write out to accessibility.csv which is the input to activitysim
@pytest.fixture(scope='module')
def prepare_module_inputs() -> None:
    """
    copy input files from sharepoint into test folder

    create unique person id in person file

    :return: None
    """
    # https://wsponlinenam.sharepoint.com/sites/US-TM2ConversionProject/Shared%20Documents/Forms/
    # AllItems.aspx?id=%2Fsites%2FUS%2DTM2ConversionProject%2FShared%20Documents%2FTask%203%20ActivitySim&viewid=7a1eaca7%2D3999%2D4d45%2D9701%2D9943cc3d6ab1
    accessibility_file = os.path.join('test', 'auto_ownership', 'data', 'accessibilities.csv')
    household_file = os.path.join('test', 'auto_ownership', 'data', 'popsyn', 'households.csv')
    person_file = os.path.join('test', 'auto_ownership', 'data', 'popsyn', 'persons.csv')
    landuse_file = os.path.join('test', 'auto_ownership', 'data', 'landuse', 'maz_data_withDensity.csv')

    test_dir = os.path.join('test', 'cdap', 'data')

    shutil.copy(accessibility_file, os.path.join(test_dir, 'accessibility.csv'))
    shutil.copy(household_file, os.path.join(test_dir, 'households.csv'))
    shutil.copy(person_file, os.path.join(test_dir, 'persons.csv'))
    shutil.copy(landuse_file, os.path.join(test_dir, 'land_use.csv'))

    # add original maz id to accessibility table
    land_use_df = pd.read_csv(
        os.path.join(test_dir, 'land_use.csv')
    )
    accessibility_df = pd.read_csv(
        os.path.join(test_dir, 'accessibility.csv')
    )

    accessibility_df = pd.merge(
        accessibility_df,
        land_use_df[['MAZ', 'MAZ_ORIGINAL']].rename(
            columns = {'MAZ' : 'mgra'}
        ),
        how = 'left',
        on = 'mgra'
    )

    accessibility_df.to_csv(
        os.path.join(test_dir, 'accessibility.csv'),
        index = False
    )
    
    # currently household file has to have these two columns, even before annotation
    # because annotate person happens before household and uses these two columns
    # TODO find a way to get around this
    ####
    household_df = pd.read_csv(
        os.path.join(test_dir, 'households.csv')
    )

    household_columns_dict = {
        'HHID' : 'household_id',
        'MAZ' : 'home_zone_id'
    }

    household_df.rename(columns = household_columns_dict, inplace = True)

    tm2_simulated_household_df = pd.read_csv(
        os.path.join(test_dir, 'householdData_3.csv')
    )
    tm2_simulated_household_df.rename(columns = {'hh_id' : 'household_id'}, inplace = True)

    household_df = pd.merge(
        household_df,
        tm2_simulated_household_df[
            ['household_id', 'autos', 'automated_vehicles', 'transponder', 'cdap_pattern', 'jtf_choice']
        ],
        how = 'inner', # tm2 is not 100% sample run
        on = 'household_id'
    )

    household_df.to_csv(
        os.path.join('test', 'cdap', 'data', 'households.csv'),
        index = False
    )

    person_df = pd.read_csv(
        os.path.join(test_dir, 'persons.csv')
    )

    person_columns_dict = {
        'HHID' : 'household_id',
        'PERID' : 'person_id'
    }

    person_df.rename(columns = person_columns_dict, inplace = True)

    tm2_simulated_person_df = pd.read_csv(
        os.path.join(test_dir, 'personData_3.csv')
    )
    tm2_simulated_person_df.rename(columns = {'hh_id' : 'household_id'}, inplace = True)

    person_df = pd.merge(
        person_df,
        tm2_simulated_person_df[
            [
                'household_id', 
                'person_id', 
                'type', 
                'value_of_time', 
                'activity_pattern',
                'imf_choice',
                'inmf_choice',
                'fp_choice',
                'reimb_pct',
                'workDCLogsum',
                'schoolDCLogsum'
            ]
        ],
        how = 'inner', # tm2 is not 100% sample run
        on = ['household_id', 'person_id']
    )

    person_df.to_csv(
        os.path.join(test_dir, 'persons.csv'),
        index = False
    )
    ####

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