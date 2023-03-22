# ActivitySim
# See full license in LICENSE.txt.
from __future__ import annotations

import logging

import pandas as pd

from activitysim.core import config, estimation, simulate, tracing, workflow

logger = logging.getLogger(__name__)


@workflow.step
def auto_ownership_simulate(
    state: workflow.State,
    households: pd.DataFrame,
    households_merged: pd.DataFrame,
):
    """
    Auto ownership is a standard model which predicts how many cars a household
    with given characteristics owns
    """
    trace_label = "auto_ownership_simulate"
    model_settings_file_name = "auto_ownership.yaml"
    model_settings = state.filesystem.read_model_settings(model_settings_file_name)
    trace_hh_id = state.settings.trace_hh_id

    estimator = estimation.manager.begin_estimation(state, "auto_ownership")
    model_spec = state.filesystem.read_model_spec(file_name=model_settings["SPEC"])
    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec, coefficients_df, estimator
    )

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    choosers = households_merged

    logger.info("Running %s with %d households", trace_label, len(choosers))

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df, model_settings)
        estimator.write_choosers(choosers)

    log_alt_losers = state.settings.log_alt_losers

    choices = simulate.simple_simulate(
        state,
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="auto_ownership",
        log_alt_losers=log_alt_losers,
        estimator=estimator,
    )

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "households", "auto_ownership")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    # no need to reindex as we used all households
    households["auto_ownership"] = choices

    state.add_table("households", households)

    tracing.print_summary(
        "auto_ownership", households.auto_ownership, value_counts=True
    )

    if trace_hh_id:
        state.tracing.trace_df(households, label="auto_ownership", warn_if_empty=True)