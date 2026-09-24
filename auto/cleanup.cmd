CLS

:: Semicolon-separated directory names to remove recursively
set "CLEANUP_NAMES=__pycache__;cache"

:: Clear configured directories in the current directory and any subdirectories
python -c "import pathlib, shutil, os, stat; f=lambda func, path, _: (os.chmod(path, stat.S_IWRITE), func(path)); cleanup_names=os.environ['CLEANUP_NAMES'].split(';'); [shutil.rmtree(p, onerror=f) for p in pathlib.Path().rglob('*') if p.is_dir() and p.name in cleanup_names]"

:: Clear any caches in the current directory and any subdirectories
uvx ruff clean
