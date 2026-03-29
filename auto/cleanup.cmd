CLS

:: Clear any __pycache__ directories in the current directory and any subdirectories
python -c "import pathlib, shutil, os, stat; f=lambda func, path, _: (os.chmod(path, stat.S_IWRITE), func(path)); [shutil.rmtree(p, onerror=f) for p in pathlib.Path().rglob('__pycache__')]"

:: Clear any caches in the current directory and any subdirectories
uvx ruff clean