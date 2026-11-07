"""Controlled local dependency fixture, NOT an installed PyPI package."""
__version__ = "2.0.0"

def read_config(text):
    key, value = text.split("=", 1)
    return {key: value}
