"""Application entry point; imports never start inference or repository execution."""
import sys
from .app_cli import main as app_main

def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "doctor":
        args[0] = "app-doctor"
    return app_main(args)
