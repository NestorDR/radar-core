:: new_uv_venv.cmd

:: $Home replaces %USERPROFILE% in PowerShell

:: Set virtual environment (venv) outside the project folder structure for reusability
set ENV_FOLDER=%USERPROFILE%\repos\VirtualEnvironments\radar-3.13.13-uv-env

:: Create with uv (universal virtualenv) the venv
:: --seed : will install pip, setuptools and wheel into the new environment, making it 100% compatible with PyCharm.
uv venv --python 3.13.13 --seed %ENV_FOLDER%

:: Activate
call %ENV_FOLDER%\Scripts\activate.bat

:: Change directory to the project folder and ...
cd %OneDriveConsumer%\Radar\radar-core\

:: ... build a bridge between local source code and the venv, making your project "importable" from anywhere within that environment
::  as if it were a third-party package you downloaded from the internet.
uv pip install -e .