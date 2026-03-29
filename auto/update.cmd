@echo off
cls

:: Self-actualization
uv self update

::Set python environment
set ENV_FOLDER=%USERPROFILE%\repos\VirtualEnvironments\radar-3.13.11-uv-env
call %ENV_FOLDER%\Scripts\activate.bat

uv pip uninstall talib

:: Update the project's lock file to the latest allowed versions according to the restrictions in pyproject.toml
uv lock --upgrade

:: Sync the virtual environment to the project's lock file (uv.lock)
:: Use --group dev to include the dev dependencies
:: Use --active to target the active environment
uv sync --group dev --active
:: Im Pycharm: File → Reload from Disk (Ctrl+F5)
::             Python Interpreter → 🔄 (Refresh icon)

:: Update TA-lib
uv pip install --no-cache-dir https://github.com/cgohlke/talib-build/releases/download/v0.6.4/ta_lib-0.6.4-cp313-cp313-win_amd64.whl

::Run python script freeze: to creates a list of all the installed packages in the virtual environment, along with their versions.
:: Previously %ENV_FOLDER%\Scripts\pip3.exe freeze > %PROJECT_FOLDER%\requirements.txt
@echo Freezing requirements
uv pip freeze > requirements.lock

::Return to initial folder
::cd %AUTO_FOLDER%
