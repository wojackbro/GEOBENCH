"""
Path configuration for CityLens data. Set CITYLENS_DATA_ROOT to the directory
that contains Benchmark/, Dataset/, satellite_image/, street_view_image/
(e.g. the extracted CityLens-Data folder from Hugging Face).
"""
import os

# Root directory for CityLens data (extracted CityLens-Data zip)
_CITYLENS_ROOT = os.environ.get("CITYLENS_DATA_ROOT", "")
if not _CITYLENS_ROOT:
    # Default: assume we're in geo_ai1/CityLens and data is in geo_ai1/data/CityLens-Data
    _here = os.path.dirname(os.path.abspath(__file__))
    _CITYLENS_ROOT = os.path.join(_here, "..", "..", "data", "CityLens-Data")

DATA_ROOT = os.path.normpath(_CITYLENS_ROOT)


def benchmark_path(task_name: str, city: str) -> str:
    """Path to task JSON for global tasks (e.g. gdp, pop)."""
    return os.path.join(DATA_ROOT, "Benchmark", f"{task_name}_{city}.json")


def results_path(task_name: str, city: str, model_name: str, prompt_type: str) -> str:
    """Path to save/load model response JSON."""
    safe_model = model_name.replace("/", "_")
    return os.path.join(DATA_ROOT, "Results", f"{task_name}_{city}_{safe_model}_{prompt_type}.json")


def summary_csv_path() -> str:
    """Path to summary metrics CSV."""
    return os.path.join(DATA_ROOT, "Results", "summary.csv")


def url_file_path() -> str:
    """Path to CSV mapping image_name -> image_url (for API-based eval)."""
    return os.path.join(DATA_ROOT, "Benchmark", "image_urls.csv")
