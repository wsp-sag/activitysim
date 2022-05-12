# ActivitySim
# See full license in LICENSE.txt.
import logging

import numpy as np
import pandas as pd
import os
from activitysim.core.interaction_simulate import interaction_simulate

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject
from activitysim.core import expressions

from .util import estimation

from .util.overlap import hh_time_window_overlap
from .util.tour_frequency import process_joint_tours

logger = logging.getLogger(__name__)


@inject.step()
def joint_tour_frequency_composition(
        households,
        persons,
        chunk_size,
        trace_hh_id):
    """
    This model predicts the frequency and composition of fully joint tours.
    """
   
    trace_label = 'joint_tour_frequency_composition'
    model_settings_file_name = 'joint_tour_frequency_composition.yaml'

    model_settings = config.read_model_settings(model_settings_file_name)

    alts_jtf = simulate.read_model_alts('joint_tour_frequency_composition_alternatives.csv', set_index='alt')

    # - only interested in households with more than one cdap travel_active person and
    # - at least one non-preschooler
    households = households.to_frame()
    choosers = households[households.participates_in_jtf_model].copy()

    # - only interested in persons in choosers households
    persons = persons.to_frame()
    persons = persons[persons.household_id.isin(choosers.index)]
    
    logger.info("Running %s with %d households", trace_label, len(choosers))

    # alt preprocessor
    alt_preprocessor_settings = model_settings.get('ALTS_PREPROCESSOR', None)
    if alt_preprocessor_settings:

        locals_dict = {}

        alts_jtf = alts_jtf.copy()

        expressions.assign_columns(
            df=alts_jtf,
            model_settings=alt_preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    # - preprocessor
    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:

        locals_dict = {
            'persons': persons,
            'hh_time_window_overlap': hh_time_window_overlap
        }

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_dict,
            trace_label=trace_label)

    choosers.to_csv(os.path.join('test', 'joint_tours', 'output', 'joint_tours_choosers.csv'), index = False)
    alts_jtf.to_csv(os.path.join('test', 'joint_tours', 'output', 'joint_tours_alts.csv'), index = False)

    estimator = estimation.manager.begin_estimation('joint_tour_frequency_composition')
    
    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df, estimator)

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    if estimator:
        estimator.write_spec(model_settings)
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)
        estimator.write_alternatives(alts)

        assert choosers.index.name == 'household_id'
        assert 'household_id' not in choosers.columns
        choosers['household_id'] = choosers.index

        estimator.set_chooser_id(choosers.index.name)

    choices = interaction_simulate(
        choosers=choosers,
        alternatives=alts_jtf,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name=trace_label,
        estimator=estimator)

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, 'households', 'joint_tour_frequency_composition')
        estimator.write_override_choices(choices)
        estimator.end_estimation()

