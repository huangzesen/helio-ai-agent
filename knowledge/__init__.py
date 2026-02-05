"""Knowledge base for spacecraft datasets and HAPI integration."""

from .catalog import (
    SPACECRAFT,
    list_spacecraft,
    list_instruments,
    get_datasets,
    match_spacecraft,
    match_instrument,
    search_by_keywords,
)
from .hapi_client import (
    get_dataset_info,
    list_parameters,
)
