"""Controlled local dependency fixture, NOT an installed PyPI package."""
__version__ = "1.4.0"

def load_config(text):
    key, value = text.split("=", 1)
    return {key: value}
