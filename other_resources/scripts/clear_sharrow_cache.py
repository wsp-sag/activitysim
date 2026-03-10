#!/usr/bin/env -S uv run --script --no-project
#
# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = [
#   "platformdirs",
# ]
# ///

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import platformdirs


def clear_sharrow_cache(archive: bool = True):
    """Disable all files in the sharrow cache directory.

    Parameters
    ----------
    archive : bool, optional
        If True, move the files to an 'archive' subdirectory instead of deleting them,
        by default True.
    """
    main_dir = Path(platformdirs.user_cache_dir(appname="ActivitySim"))
    print(f"Scanning ActivitySim cache directory for sharrow files: {main_dir}")
    sharrow_cache_dirs = main_dir.glob("sharrow-*-numba-*")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # define the archive path if archiving is enabled
    if archive:
        archive_path = Path(
            platformdirs.user_cache_dir(appname="ActivitySim")
        ).joinpath(f"archive-{timestamp}")
    else:
        archive_path = None

    for sharrow_cache_dir in sharrow_cache_dirs:
        if sharrow_cache_dir.is_dir():
            print(f"Clearing sharrow cache directory: {sharrow_cache_dir}")
            if archive:
                archive_path.mkdir(parents=True, exist_ok=True)
                shutil.move(
                    str(sharrow_cache_dir), str(archive_path / sharrow_cache_dir.name)
                )
            else:
                shutil.rmtree(str(sharrow_cache_dir))
    else:
        print("No sharrow cache directories found.")


if __name__ == "__main__":
    clear_sharrow_cache(archive=True)
