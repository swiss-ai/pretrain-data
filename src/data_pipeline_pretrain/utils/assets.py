import os
import importlib.resources

ASSETS_PATH = os.path.abspath(
    importlib.resources.files(__package__.split(".")[0]).joinpath("assets")
)
