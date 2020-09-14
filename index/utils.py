from pathlib import Path

def createDirIfNotExists(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)


