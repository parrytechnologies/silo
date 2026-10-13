"""SILO — Search and Indexing, Local-Only: CPU-only, air-gapped document search."""

from silo._offline import enforce_offline

# Must run before anything imports huggingface_hub / transformers: those libraries
# read their offline switches once, at import time.
enforce_offline()

__version__ = "0.3.0"
