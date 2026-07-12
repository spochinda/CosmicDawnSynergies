from importlib.metadata import version

try:
    __version__ = version("CosmicDawnSynergies")
except Exception:
    __version__ = "unknown"
