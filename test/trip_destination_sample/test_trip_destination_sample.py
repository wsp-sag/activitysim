import logging
import pytest
import os
import shutil
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import orca
import itertools

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
    return 'trip_destination_sample'


# Used by conftest.py initialize_pipeline method
@pytest.fixture(scope='module')
def tables(prepare_module_inputs) -> dict[str, str]:
    """
    A pytest fixture that returns the "mock" tables to build pipeline dataframes. The
    key-value pair is the name of the table and the index column.
    :return: dict
    """
    return {
        'land_use': 'zone_id',
        'persons': 'person_id',
        'households': 'household_id',
        'accessibility': 'maz',
        'tours': 'tour_id',
        'trips': 'trip_id',
        'size_terms': 'segment'
    }


# Used by conftest.py initialize_pipeline method
# Set to true if you need to read skims into the pipelinfinitialize_network_lose
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
@pytest.mark.skipif(os.path.isfile('test/trip_destination_sample/output/pipeline.h5'), reason = "no need to recreate pipeline store if already exist")
def test_prepare_input_pipeline(initialize_pipeline: pipeline.Pipeline, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    # run model step
    pipeline.run(models=[
        'initialize_landuse',
        'initialize_households'
        ]
    )

    output_dir = os.path.join('test', 'trip_destination_sample', 'output')

    # save the updated pipeline tables
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


#@pytest.mark.skip
def test_trip_destination_sample(reconnect_pipeline: pipeline.Pipeline, caplog):
    # Run summarize model
    caplog.set_level(logging.INFO)

    # run model step
    pipeline.run(models=[
        'trip_destination'
        ],
        resume_after = 'initialize_households'
    )
    
    # get the updated trips data
    trips_df = pipeline.get_table('trips')
    trips_df.to_csv('test/trip_destination_sample/output/trips_after_destination_choice.csv')


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
    test_dir = os.path.join('test', 'trip_destination_sample', 'data')
    
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
    land_use_df['zone_id'] = land_use_df['maz']

    land_use_df.to_csv(
        os.path.join(test_dir, 'land_use.csv'),
        index = False
    )
    # land_use_df[['maz', 'maz_county_based']]
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
        'MAZ' : 'maz_county_based'
    }

    household_df.rename(columns = household_columns_dict, inplace = True)

    # maz in the households.csv is the county based maz, get the sequential maz for home zone
    household_df = pd.merge(
        household_df,
        land_use_df[['maz', 'maz_county_based']],
        how='left',
        on='maz_county_based'
    )
    household_df.rename(columns = {'maz' : 'home_zone_id'}, inplace = True)

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

    person_df['WorkLocation'] = np.where(person_df['WorkLocation'] == 99999, -1, person_df['WorkLocation'])
    person_df['SchoolLocation'] = np.where(person_df['SchoolLocation'] == 88888, -1, person_df['SchoolLocation'])
    person_df['WorkLocation'] = np.where(person_df['WorkLocation'] == 0, -1, person_df['WorkLocation'])
    person_df['SchoolLocation'] = np.where(person_df['SchoolLocation'] == 0, -1, person_df['SchoolLocation'])

    ## get tour data from tm2 output
    tm2_simulated_indiv_tour_df = pd.read_csv(
        os.path.join(test_dir, 'tm2_outputs', 'indivTourData_1.csv')
    )

    tm2_simulated_joint_tour_df = pd.read_csv(
        os.path.join(test_dir, 'tm2_outputs', 'jointTourData_1.csv')
    )

    tm2_simulated_indiv_tour_df.drop(tm2_simulated_indiv_tour_df.filter(regex='util').columns, axis=1, inplace=True)
    tm2_simulated_indiv_tour_df.drop(tm2_simulated_indiv_tour_df.filter(regex='prob').columns, axis=1, inplace=True)

    tm2_simulated_indiv_tour_df['unique_tour_id'] = range(1, len(tm2_simulated_indiv_tour_df) + 1)


    number_of_participants = [*map(lambda x: len(x.split(" ")),tm2_simulated_joint_tour_df.tour_participants.tolist())]
    primary_participant_num = [*map(lambda x: x.split(" ")[0],tm2_simulated_joint_tour_df.tour_participants.tolist())]
    tour_participants = [*map(lambda x: x.split(" "),tm2_simulated_joint_tour_df.tour_participants.tolist())]
    tour_participants = list(itertools.chain.from_iterable(tour_participants))

    tm2_simulated_joint_tour_df['unique_tour_id'] = range(len(tm2_simulated_indiv_tour_df) + 1, len(tm2_simulated_indiv_tour_df) + len(tm2_simulated_joint_tour_df) + 1)
    tm2_simulated_joint_tour_df['composition'] = tm2_simulated_joint_tour_df['tour_composition']
    tm2_simulated_joint_tour_df['composition'] = np.where(tm2_simulated_joint_tour_df['composition'] == 1, 'adult', tm2_simulated_joint_tour_df['composition'])
    tm2_simulated_joint_tour_df['composition'] = np.where(tm2_simulated_joint_tour_df['composition'] == 2, 'children', tm2_simulated_joint_tour_df['composition'])
    tm2_simulated_joint_tour_df['composition'] = np.where(tm2_simulated_joint_tour_df['composition'] == 3, 'mixed', tm2_simulated_joint_tour_df['composition'])

    tm2_simulated_joint_tour_df['primary_participant_num'] = primary_participant_num
    tm2_simulated_joint_tour_df['number_of_participants'] = number_of_participants
    
    tm2_simulated_joint_tour_df['primary_participant_num'] = tm2_simulated_joint_tour_df['primary_participant_num'].astype(int)
    tm2_simulated_joint_tour_df['number_of_participants'] = tm2_simulated_joint_tour_df['number_of_participants'].astype(int)

    tm2_simulated_joint_tour_df = pd.merge(
        tm2_simulated_joint_tour_df, 
        person_df[['household_id', 'person_id', 'person_num']],
        how='left',
        left_on=['hh_id', 'primary_participant_num'],
        right_on=['household_id', 'person_num'],
    )

    tm2_simulated_joint_tour_df.drop(tm2_simulated_joint_tour_df.filter(regex='util').columns, axis=1, inplace=True)
    tm2_simulated_joint_tour_df.drop(tm2_simulated_joint_tour_df.filter(regex='prob').columns, axis=1, inplace=True)
    tm2_simulated_joint_tour_df.drop(['tour_composition', 'tour_participants', 'primary_participant_num', 'household_id'], axis=1, inplace=True)

    tm2_simulated_tour_df = pd.concat(
        [tm2_simulated_indiv_tour_df, tm2_simulated_joint_tour_df],
        sort = False,
        ignore_index = True
    )

    tm2_simulated_tour_df.rename(columns = {'hh_id' : 'household_id'}, inplace=True)
    
    tm2_simulated_tour_df['tour_category'] = tm2_simulated_tour_df['tour_category'].str.lower()
    tm2_simulated_tour_df['tour_category'] = np.where(tm2_simulated_tour_df.tour_category == 'individual_non_mandatory', 'non_mandatory', tm2_simulated_tour_df.tour_category)
    tm2_simulated_tour_df['tour_category'] = np.where(tm2_simulated_tour_df.tour_category == 'joint_non_mandatory', 'joint', tm2_simulated_tour_df.tour_category)
    tm2_simulated_tour_df['tour_category'] = np.where(tm2_simulated_tour_df.tour_category == 'at_work', 'atwork', tm2_simulated_tour_df.tour_category)

    tm2_simulated_tour_df['tour_purpose'] = np.where(tm2_simulated_tour_df.tour_purpose == 'Work-Based', 'atwork', tm2_simulated_tour_df.tour_purpose)
    tm2_simulated_tour_df['tour_purpose'] = tm2_simulated_tour_df['tour_purpose'].str.lower()
    
    tm2_simulated_tour_df['tour_type'] = tm2_simulated_tour_df['tour_purpose']
    tm2_simulated_tour_df['tour_type'] = np.where(tm2_simulated_tour_df.tour_type == 'university', 'univ', tm2_simulated_tour_df.tour_type)
    tm2_simulated_tour_df['tour_type'] = np.where(tm2_simulated_tour_df.tour_type == 'escort', 'escort', tm2_simulated_tour_df.tour_type)
    tm2_simulated_tour_df['tour_type'] = np.where(tm2_simulated_tour_df.tour_type == 'shop', 'shopping', tm2_simulated_tour_df.tour_type)
    tm2_simulated_tour_df['tour_type'] = np.where(tm2_simulated_tour_df.tour_type == 'eating out', 'eatout', tm2_simulated_tour_df.tour_type)
    tm2_simulated_tour_df['tour_type'] = np.where(tm2_simulated_tour_df.tour_type == 'discretionary', 'othdiscr', tm2_simulated_tour_df.tour_type)
    tm2_simulated_tour_df['tour_type'] = np.where(tm2_simulated_tour_df.tour_type == 'maintenance', 'othmaint', tm2_simulated_tour_df.tour_type)
    tm2_simulated_tour_df['tour_type'] = np.where(tm2_simulated_tour_df.tour_type == 'visiting', 'social', tm2_simulated_tour_df.tour_type)

    tm2_simulated_tour_df['origin'] = tm2_simulated_tour_df['orig_mgra']
    tm2_simulated_tour_df['destination'] = tm2_simulated_tour_df['dest_mgra']

    ## get trip data from tm2 output
    tm2_simulated_indiv_trip_df = pd.read_csv(
        os.path.join(test_dir, 'tm2_outputs', 'indivTripData_1.csv')
    )
    tm2_simulated_joint_trip_df = pd.read_csv(
        os.path.join(test_dir, 'tm2_outputs', 'jointTripData_1.csv')
    )

    tm2_simulated_trip_df = pd.concat(
        [tm2_simulated_indiv_trip_df, tm2_simulated_joint_trip_df],
        sort = False,
        ignore_index = True
    )

    tm2_simulated_trip_df.rename(columns = {'hh_id' : 'household_id'}, inplace=True)

    tm2_simulated_trip_df['trip_id'] = range(1, len(tm2_simulated_trip_df) + 1)

    tm2_simulated_trip_df['purpose'] = tm2_simulated_trip_df['dest_purpose'].str.lower()

    tm2_simulated_trip_df.rename(
        columns = {
            'orig_mgra': 'origin',
            'dest_mgra': 'destination'
        },
        inplace=True
    )

    tm2_simulated_trip_df['tour_purpose'] = tm2_simulated_trip_df['tour_purpose'].str.lower()
    tm2_simulated_trip_df['primary_purpose'] = tm2_simulated_trip_df['tour_purpose']
    tm2_simulated_trip_df['primary_purpose'] = np.where(tm2_simulated_trip_df.primary_purpose == 'university', 'univ', tm2_simulated_trip_df.primary_purpose)
    tm2_simulated_trip_df['primary_purpose'] = np.where(tm2_simulated_trip_df.primary_purpose == 'escort', 'escort', tm2_simulated_trip_df.primary_purpose)
    tm2_simulated_trip_df['primary_purpose'] = np.where(tm2_simulated_trip_df.primary_purpose == 'shop', 'shopping', tm2_simulated_trip_df.primary_purpose)
    tm2_simulated_trip_df['primary_purpose'] = np.where(tm2_simulated_trip_df.primary_purpose == 'eating out', 'eatout', tm2_simulated_trip_df.primary_purpose)
    tm2_simulated_trip_df['primary_purpose'] = np.where(tm2_simulated_trip_df.primary_purpose == 'discretionary', 'othdiscr', tm2_simulated_trip_df.primary_purpose)
    tm2_simulated_trip_df['primary_purpose'] = np.where(tm2_simulated_trip_df.primary_purpose == 'maintenance', 'othmaint', tm2_simulated_trip_df.primary_purpose)
    tm2_simulated_trip_df['primary_purpose'] = np.where(tm2_simulated_trip_df.primary_purpose == 'visiting', 'social', tm2_simulated_trip_df.primary_purpose)
    tm2_simulated_trip_df['primary_purpose'] = np.where(tm2_simulated_trip_df.primary_purpose == 'work-based', 'atwork', tm2_simulated_trip_df.primary_purpose)

    tm2_simulated_trip_df['orig_purpose'] = tm2_simulated_trip_df['orig_purpose'].str.lower()
    tm2_simulated_trip_df['dest_purpose'] = tm2_simulated_trip_df['dest_purpose'].str.lower()
    tm2_simulated_trip_df['depart'] = tm2_simulated_trip_df['stop_period']

    tm2_simulated_trip_df = pd.merge(
        tm2_simulated_trip_df,
        tm2_simulated_tour_df[['household_id', 'person_id', 'tour_id', 'tour_purpose', 'unique_tour_id']],
        how='left',
        on=['household_id', 'person_id', 'tour_id', 'tour_purpose']
    )

    # drop tour id and rename unique_tour_id to tour_id
    tm2_simulated_tour_df.drop(['tour_id'], axis=1, inplace=True)
    tm2_simulated_tour_df.rename(columns = {'unique_tour_id' : 'tour_id'}, inplace=True)

    tm2_simulated_trip_df.drop(['tour_id'], axis=1, inplace=True)
    tm2_simulated_trip_df.rename(columns = {'unique_tour_id' : 'tour_id'}, inplace=True)

    tm2_simulated_trip_df['outbound'] = tm2_simulated_trip_df['inbound'] == 0

    def _process_trips(trips_df): 
        trips = trips_df.copy()

        grouped = trips.groupby(['person_id', 'tour_id', 'inbound'])
        trips['trip_num'] = grouped.cumcount() + 1
        trips['trip_count'] = trips['trip_num'] + grouped.cumcount(ascending=False)

        return trips

    trips_df = _process_trips(tm2_simulated_trip_df)

    def _process_tours(tours_df, parent_col): 
        tours = tours_df.copy()
        
        grouped = tours.groupby([parent_col, 'tour_type'])
        tours['tour_type_num'] = grouped.cumcount() + 1
        tours['tour_type_count'] = tours['tour_type_num'] + grouped.cumcount(ascending=False)
        
        grouped = tours.groupby(parent_col)
        tours['tour_num'] = grouped.cumcount() + 1
        tours['tour_count'] = tours['tour_num'] + grouped.cumcount(ascending=False)
        
        return tours

    mandatory_tours = tm2_simulated_tour_df[tm2_simulated_tour_df.tour_category == 'mandatory']
    non_mandatory_tours = tm2_simulated_tour_df[tm2_simulated_tour_df.tour_category == 'non_mandatory']
    atwork_tours = tm2_simulated_tour_df[tm2_simulated_tour_df.tour_category == 'atwork']
    joint_tours = tm2_simulated_tour_df[tm2_simulated_tour_df.tour_category == 'joint']

    mandatory_tours = _process_tours(mandatory_tours, parent_col='person_id')
    non_mandatory_tours = _process_tours(non_mandatory_tours, parent_col='person_id')
    joint_tours = _process_tours(joint_tours, parent_col='household_id')
    
    # add parent (mandatory) tour id for atwork tours
    atwork_tours = pd.merge(
        atwork_tours, 
        mandatory_tours[['tour_id', 'household_id', 'person_id', 'destination']].rename(columns = {'tour_id' : 'parent_tour_id', 'destination' : 'origin'}),
        how='left',
        on=['household_id', 'person_id', 'origin']
    )
    atwork_tours.drop_duplicates(subset=['tour_id'], inplace=True)

    atwork_tours = _process_tours(atwork_tours, parent_col='parent_tour_id')

    tours_df = pd.concat(
        [mandatory_tours, joint_tours, non_mandatory_tours, atwork_tours],
        ignore_index = True
    )
    
    # filter out auto tours only
    tours_df = tours_df[tours_df.tour_mode.isin([1, 2, 3, 4, 5, 6, 7, 8])]
    trips_df = trips_df[trips_df.tour_mode.isin([1, 2, 3, 4, 5, 6, 7, 8])]
    
    # filter work tours/trips only
    tours_df = tours_df[tours_df.tour_type == 'work']
    trips_df = trips_df[trips_df.primary_purpose == 'work']

    trips_df = trips_df.head(10000)
    
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