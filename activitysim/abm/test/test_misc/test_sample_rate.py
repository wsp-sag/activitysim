# ActivitySim
# See full license in LICENSE.txt.
import pandas as pd

import pytest

from activitysim.core import workflow


@pytest.fixture(scope="session")
def example_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("example")
    config_dir = root / "configs"
    config_dir.mkdir()

    data_dir = root / "data"
    data_dir.mkdir()

    return root


@pytest.fixture(scope="module")
def state(example_root) -> workflow.State:

    settings = """
        input_table_list:
            - tablename: households
              filename: households.csv
              index_col: household_id
        
        households_sample_size: 2
        """

    settings_file = example_root / "configs" / "settings.yaml"
    settings_file.write_text(settings)

    households = pd.DataFrame(
        {
            "household_id": range(1, 100001),
            "home_zone_id": [1, 2] * 50000,
        }
    )
    households.to_csv(example_root / "data" / "households.csv", index=False)

    state = workflow.State.make_default(example_root)

    return state


def test_sample_rate_calculation(state):
    households_df = state.get_dataframe("households")
    sample_rate = households_df["sample_rate"].iloc[0]
    assert (
        sample_rate == 0.00002
    ), f"Expected sample rate of 0.00002, but got {sample_rate}"
