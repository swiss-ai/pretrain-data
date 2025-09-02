from pathlib import Path


def list_files(dir_path, recursive=False):
    path = Path(dir_path)
    return [f for f in path.glob("**/*" if recursive else "*") if f.is_file()]


def list_folders(dir_path):
    path = Path(dir_path)
    return [f for f in path.glob("*") if f.is_dir()]
