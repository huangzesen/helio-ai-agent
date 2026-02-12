"""
Data fetcher — pulls timeseries from CDAWeb into pandas DataFrames.

Downloads CDF files from CDAWeb's REST API and reads them with cdflib.
Errors propagate directly so the agent can learn from failures
(e.g., virtual parameters that exist in Master CDF metadata but
not in actual data files).
"""

import logging

logger = logging.getLogger("helio-agent")


def fetch_data(
    dataset_id: str,
    parameter_id: str,
    time_min: str,
    time_max: str,
) -> dict:
    """Fetch timeseries data from CDAWeb via CDF file download.

    Args:
        dataset_id: CDAWeb dataset ID (e.g., "AC_H2_MFI").
        parameter_id: Parameter name (e.g., "BGSEc").
        time_min: ISO start time (e.g., "2024-01-15T00:00:00Z").
        time_max: ISO end time (e.g., "2024-01-16T00:00:00Z").

    Returns:
        Dict with keys: data (DataFrame), units, description, fill_value.

    Raises:
        ValueError: If no data is available or parameter not found.
        requests.HTTPError: If a download fails.
    """
    from data_ops.fetch_cdf import fetch_cdf_data
    return fetch_cdf_data(dataset_id, parameter_id, time_min, time_max)
