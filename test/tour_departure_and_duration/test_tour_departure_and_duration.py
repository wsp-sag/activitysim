import logging
import pytest
import os
import shutil
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import orca

# import models is necessary to initalize the model steps with orca
from activitysim.abm import models
from activitysim.abm.tables.persons import persons_merged
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
    return 'tour_departure_and_duration'


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope='module')
def tables(prepare_module_inputs) -> dict[str, str]:
    """
    A pytest fixture that returns the "mock" tables to build pipeline dataframes. The
    key-value pair is the name of the table and the index column.
    :return: dict
    """
    return {
        'land_use': 'maz',
        'persons': 'person_id',
        'households': 'household_id',
        'accessibility': 'maz',
        'tours': 'tour_id',
        'trips': 'trip_id'
    }


# Used by conftest.py initialize_pipeline method
# Set to true if you need to read skims into the pipeline
@pytest.fixture(scope='module')
def initialize_network_los() -> bool:
    """
    A pytest boolean fixture indicating whether network skims should be read from the
    fixtures test data folder.
    :return: boolcls
    """
    return True


@pytest.fixture(scope='module')
def load_checkpoint() -> bool:
    """
    checkpoint to be loaded from the pipeline when reconnecting. 
    """
    return 'initialize_households'


# make a reconnect_pipeline internal to test module
@pytest.mark.skipif(os.path.isfile('test/tour_departure_and_duration/output/pipeline.h5'), reason = "no need to recreate pipeline store if already exist")
def test_prepare_input_pipeline(initialize_pipeline: pipeline.Pipeline, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    data_dir = os.path.join('test', 'tour_departure_and_duration', 'data')
    output_dir = os.path.join('test', 'tour_departure_and_duration', 'output')

    joint_tour_participants = pd.read_csv(os.path.join(data_dir, 'joint_tour_participants.csv'))
    orca.add_table('joint_tour_participants', joint_tour_participants)

    # run model step
    pipeline.run(models=[
        'initialize_landuse',
        'initialize_households'
        ]
    )

    # # save the pipeline tables
    person_df = pipeline.get_table('persons')
    person_df.to_csv(os.path.join(output_dir, 'person.csv'))

    household_df = pipeline.get_table('households')
    household_df.to_csv(os.path.join(output_dir, 'household.csv'))

    land_use_df = pipeline.get_table('land_use')
    land_use_df.to_csv(os.path.join(output_dir, 'land_use.csv'))

    accessibility_df = pipeline.get_table('accessibility')
    accessibility_df.to_csv(os.path.join(output_dir, 'accessibility.csv'))

    tours_df = pipeline.get_table('tours')
    tours_df.to_csv(os.path.join(output_dir, 'tours.csv'))

    trips_df = pipeline.get_table('trips')
    trips_df.to_csv(os.path.join(output_dir, 'trips.csv'))

    pipeline.close_pipeline()

# Test Order:
    # mandatory_tour_scheduling
    # non_mandatory_tour_scheduling
    # joint_tour_scheduling
    # atwork_subtour_scheduling

#@pytest.mark.skip
def test_mandatory_tour_scheduing(reconnect_pipeline: pipeline.Pipeline, caplog):
    
    caplog.set_level(logging.INFO)

    pipeline.run(models=['mandatory_tour_scheduling'],
        resume_after = 'initialize_households'
    )

    pipeline.close_pipeline()


@pytest.mark.skip
def test_nonmandatory_tour_scheduing(reconnect_pipeline: pipeline.Pipeline, caplog):
    
    caplog.set_level(logging.INFO)

    pipeline.run(models=['non_mandatory_tour_scheduling'],
        resume_after = 'mandatory_tour_scheduling'
    )

    pipeline.close_pipeline()


@pytest.mark.skip
def test_joint_tour_scheduing(reconnect_pipeline: pipeline.Pipeline, caplog):
    
    caplog.set_level(logging.INFO)

    pipeline.run(models=['joint_tour_scheduling'],
        resume_after = 'non_mandatory_tour_scheduling'
    )

    pipeline.close_pipeline()


@pytest.mark.skip
def test_atwork_tour_scheduing(reconnect_pipeline: pipeline.Pipeline, caplog):
    
    caplog.set_level(logging.INFO)

    pipeline.run(models=['atwork_subtour_scheduling'],
        resume_after = 'joint_tour_scheduling'
    )

    pipeline.close_pipeline()


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
    test_dir = os.path.join('test', 'tour_departure_and_duration', 'data')
    
    accessibility_file = os.path.join(test_dir, 'accessibilities.csv')
    household_file = os.path.join(test_dir, 'popsyn', 'households.csv')
    person_file = os.path.join(test_dir, 'popsyn', 'persons.csv')
    landuse_file = os.path.join(test_dir, 'landuse', 'maz_data_withDensity.csv')
    
    shutil.copy(accessibility_file, os.path.join(test_dir, 'accessibility.csv'))
    shutil.copy(household_file, os.path.join(test_dir, 'households.csv'))
    shutil.copy(person_file, os.path.join(test_dir, 'persons.csv'))
    shutil.copy(landuse_file, os.path.join(test_dir, 'land_use.csv'))

    # add original maz id to accessibility table
    land_use_df = pd.read_csv(
        os.path.join(test_dir, 'land_use.csv')
    )

    land_use_df.rename(columns = {'MAZ': 'maz', 'MAZ_ORIGINAL': 'maz_county_based'}, inplace = True)

    land_use_df.to_csv(
        os.path.join(test_dir, 'land_use.csv'),
        index = False
    )

    accessibility_df = pd.read_csv(
        os.path.join(test_dir, 'accessibility.csv')
    )

    accessibility_df['maz'] = accessibility_df['mgra']

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
        'TAZ' : 'taz',
        'MAZ' : 'home_zone_id'
    }

    household_df.rename(columns = household_columns_dict, inplace = True)

    tm2_simulated_household_df = pd.read_csv(
        os.path.join(test_dir, 'tm2_outputs', 'householdData_1.csv')
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

    person_df = pd.read_csv(
        os.path.join(test_dir, 'persons.csv')
    )

    person_columns_dict = {
        'HHID' : 'household_id',
        'PERID' : 'person_id'
    }

    person_df.rename(columns = person_columns_dict, inplace = True)

    tm2_simulated_person_df = pd.read_csv(
        os.path.join(test_dir, 'tm2_outputs', 'personData_1.csv')
    )
    tm2_simulated_person_df.rename(columns = {'hh_id' : 'household_id'}, inplace = True)

    person_df = pd.merge(
        person_df,
        tm2_simulated_person_df[
            [
                'household_id', 
                'person_id', 
                'person_num',
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

    # get tm2 simulated workplace and school location results
    tm2_simulated_wsloc_df = pd.read_csv(
        os.path.join(test_dir, 'tm2_outputs', 'wsLocResults_1.csv')
    )
    tm2_simulated_wsloc_df.rename(columns = {'HHID' : 'household_id', 'PersonID' : 'person_id'}, inplace = True)

    person_df = pd.merge(
        person_df,
        tm2_simulated_wsloc_df[
            [
                'household_id', 
                'person_id', 
                'WorkLocation',
                'WorkLocationLogsum',  # this is the same as `workDCLogsum` in tm2 person output
                'SchoolLocation',
                'SchoolLocationLogsum'  # this is the same as `schoolDCLogsum` in tm2 person output
            ]
        ],
        how = 'inner', # ctramp might not be 100% sample run
        on = ['household_id', 'person_id']
    )

    ## get tour data from tm2 output
    tm2_simulated_indiv_tour_df = pd.read_csv(
        os.path.join(test_dir, 'tm2_outputs', 'indivTourData_1.csv')
    )

    # tm2_simulated_joint_tour_df = pd.read_csv(
    #     os.path.join(test_dir, 'tm2_outputs', 'jointTourData_1.csv')
    # )

    # using joint tours data after running joint tour participation model 
    tm2_simulated_joint_tour_df = pd.read_csv(
        os.path.join(test_dir, 'joint_tours.csv')
    )

    # tours_df = pd.concat(
    #     [tm2_simulated_indiv_tour_df, tm2_simulated_joint_tour_df],
    #     sort = False,
    #     ignore_index = True
    # )
    tours_df = tm2_simulated_indiv_tour_df.copy()

    tours_df.rename(columns = {'hh_id' : 'household_id'}, inplace=True)

    #tours_df['unique_tour_id'] = range(1, len(tours_df) + 1)
    tours_df['tour_category'] = tours_df['tour_category'].str.lower()
    tours_df['tour_category'] = np.where(tours_df.tour_category == 'individual_non_mandatory', 'non_mandatory', tours_df.tour_category)
    tours_df['tour_category'] = np.where(tours_df.tour_category == 'at_work', 'atwork', tours_df.tour_category)
    tours_df['tour_type'] = tours_df['tour_purpose'].str.lower()
    tours_df['origin'] = tours_df['orig_mgra']
    tours_df['destination'] = tours_df['dest_mgra']

    def _process_tours(tours_df, parent_col): 
        tours = tours_df.copy()
        
        grouped = tours.groupby([parent_col, 'tour_type'])
        tours['tour_type_num'] = grouped.cumcount() + 1
        tours['tour_type_count'] = tours['tour_type_num'] + grouped.cumcount(ascending=False)
        
        grouped = tours.groupby(parent_col)
        tours['tour_num'] = grouped.cumcount() + 1
        tours['tour_count'] = tours['tour_num'] + grouped.cumcount(ascending=False)
        
        return tours

    mandatory_tours = tours_df[tours_df.tour_category == 'mandatory']
    non_mandatory_tours = tours_df[tours_df.tour_category == 'non_mandatory']
    atwork_tours = tours_df[tours_df.tour_category == 'atwork']
    joint_tours = tm2_simulated_joint_tour_df.copy()

    mandatory_tours = _process_tours(mandatory_tours, parent_col='person_id')
    non_mandatory_tours = _process_tours(non_mandatory_tours, parent_col='person_id')

    tours_df = pd.concat(
        [mandatory_tours, non_mandatory_tours, joint_tours, atwork_tours],
        ignore_index = True
    )

    ## get trip data from tm2 output
    tm2_simulated_indiv_trip_df = pd.read_csv(
        os.path.join(test_dir, 'tm2_outputs', 'indivTripData_1.csv')
    )
    tm2_simulated_joint_trip_df = pd.read_csv(
        os.path.join(test_dir, 'tm2_outputs', 'jointTripData_1.csv')
    )

    trips_df = pd.concat(
        [tm2_simulated_indiv_trip_df, tm2_simulated_joint_trip_df],
        sort = False,
        ignore_index = True
    )

    trips_df.rename(columns = {'hh_id' : 'household_id'}, inplace=True)

    trips_df['trip_id'] = range(1, len(trips_df) + 1)

    # trips_df = pd.merge(
    #     trips_df,
    #     tours_df[['household_id', 'person_id', 'tour_id', 'tour_purpose', 'unique_tour_id', 'start_period', 'end_period']],
    #     how='left',
    #     on=['household_id', 'person_id', 'tour_id', 'tour_purpose']
    # )

    # drop tour id and rename unique_tour_id to tour_id
    #tours_df.drop(['tour_id'], axis=1, inplace=True)
    #tours_df.rename(columns = {'unique_tour_id' : 'tour_id'}, inplace=True)

    # trips_df.drop(['tour_id'], axis=1, inplace=True)
    # trips_df.rename(
    #     columns = {
    #         'unique_tour_id' : 'tour_id',
    #         'orig_mgra': 'orig_maz',
    #         'dest_mgra': 'dest_maz',
    #         'start_period': 'tour_start_period',
    #         'end_period': 'tour_end_period'
    #     },
    #     inplace=True
    # )

    #take sample data
    household_df = household_df.sample(10000, random_state=1)
    person_df = person_df[person_df.household_id.isin(household_df.household_id)]
    tours_df = tours_df[tours_df.household_id.isin(household_df.household_id)]
    trips_df = trips_df[trips_df.household_id.isin(household_df.household_id)]

    household_df.to_csv(
        os.path.join(test_dir, 'households.csv'),
        index = False
    )

    person_df.to_csv(
        os.path.join(test_dir, 'persons.csv'),
        index = False
    )

    tours_df.to_csv(
        os.path.join(test_dir, 'tours.csv'),
        index = False
    )
    
    trips_df.to_csv(
        os.path.join(test_dir, 'trips.csv'),
        index = False
    )


def create_summary(input_df, key, out_col = "Share") -> pd.DataFrame:
    """
    Create summary for the input data. 
    1. group input data by the "key" column
    2. calculate the percent of input data records in each "key" category. 

    :return: pd.DataFrame
    """

    out_df = input_df.groupby(key).size().reset_index(name='Count')
    out_df[out_col] = round(out_df["Count"]/out_df["Count"].sum(), 4)
    
    return out_df[[key, out_col]]


def cosine_similarity(a, b): 
    """
    Computes cosine similarity between two vectors.
    
    Cosine similarity is used here as a metric to measure similarity between two sequence of numbers.
    Two sequence of numbers are represented as vectors (in a multi-dimensional space) and cosine similiarity is defined as the cosine of the angle between them
    i.e., dot products of the vectors divided by the product of their lengths. 

    :return: 
    """
    
    return dot(a, b)/(norm(a)*norm(b))


def compare_simulated_against_target(target_df: pd.DataFrame, simulated_df: pd.DataFrame, target_key: str, simulated_key:str) -> bool:
    """
    compares the simulated and target results by computing the cosine similarity between them. 

    :return:
    """
    
    merged_df = pd.merge(target_df, simulated_df, left_on=target_key, right_on=simulated_key, how="outer")
    merged_df = merged_df.fillna(0)

    logger.info("simulated vs target share:\n%s" % merged_df)

    similarity_value = cosine_similarity(merged_df["Target_Share"].tolist(), merged_df["Simulated_Share"].tolist())

    logger.info("cosine similarity:\n%s" % similarity_value)

    return similarity_value