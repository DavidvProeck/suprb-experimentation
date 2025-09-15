from pathlib import Path

def get_project_root(marker_files=(".git", "pyproject.toml", "requirements.txt")):
    """Return the path to the project root by searching upward for a marker file or folder."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if any((current / marker).exists() for marker in marker_files):
            return current
        current = current.parent
    raise FileNotFoundError(f"Could not find project root using markers: {marker_files}")


