import logging
import pytest
import os
import pandas as pd

# import models is necessary to initalize the model steps with orca
from activitysim.abm import models
from activitysim.core import pipeline, config
from activitysim.core import tracing


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope='module')
def module() -> str:
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
        #'accessibility': 'mgra'
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
    """
    target_df = pd.DataFrame()
    model_df = household_df['auto_ownership'].value_counts().to_frame()
    assert (target_df == model_df)
    """

## TODO
# fetch/prepare existing files for model inputs
# e.g. read accessibilities.csv from ctramp result, rename columns, write out to accessibility.csv which is the input to activitysim

@pytest.fixture(scope='module')
def prepare_model_inputs() -> None:
    """
    Prepare accessibility file from ctramp run into input file activitysim expects
    1. renaming columns
    2. write out table

    Prepare household, person, landuse data into format activitysim expects
    1. renaming columns
    2. write out table

    :return: None
    """
