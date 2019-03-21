from pathlib import Path

def get_suffix(file):
    return Path(file).suffix.lstrip('.')

def is_suffix(file, suffix):
    return get_suffix(file) == suffix