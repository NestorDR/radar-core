:: new_uv_venv.cmd

set ENV_FOLDER=%USERPROFILE%\repos\VirtualEnvironments\radar-3.13.13-uv-env

:: cmd
:: --seed : will install pip, setuptools and wheel into the new environment, making it 100% compatible with PyCharm.
uv venv --python 3.13.13 --seed %ENV_FOLDER%

:: powershell
:: uv venv --python 3.13.13 --seed "$HOME\repos\VirtualEnvironments\radar-3.13.13-uv-env"
:: uv venv --python 3.13.13 --seed "$env:USERPROFILE\repos\VirtualEnvironments\radar-3.13.13-uv-env"

:: activate
call %ENV_FOLDER%\Scripts\activate.bat

cd %OneDriveConsumer%\Radar\radar-core\

:: build a bridge between local source code and the venv, making your project "importable" from anywhere within that environment
::  as if it were a third-party package you downloaded from the internet.
uv pip install -e .