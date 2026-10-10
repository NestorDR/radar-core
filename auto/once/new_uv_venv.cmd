:: auto/new_uv_venv.cmd
:: Purpose: Create a new virtual environment (venv) for the project, pinned to a specific Python version, and set up the project for development.
:: Usage: auto\new_uv_venv.cmd (run from the project root directory)
@echo off
cls

:: setlocal preserves the original environment variables; endlocal restores them before exit
setlocal

:: Set virtual environment (venv) outside the project folder structure for reusability
:: e.g. to separate environment: set VIRTUAL_ENV=C:\Development\VirtualEnvs\radar-3.14.8-uv-env
set VIRTUAL_ENV=.venv

:: Create with uv (universal virtualenv) the virtual environment pinned to that exact Python version
:: --seed : will install pip, setuptools and wheel into the new environment, making it 100% compatible with PyCharm.
uv venv --python 3.14.8 --seed %VIRTUAL_ENV%

:: Activate
call %VIRTUAL_ENV%\Scripts\activate.bat

:: Ensure you are in the project folder and ...
:: cd C:\Development\Repos\radar\radar-core

:: ... build a bridge between local source code and the venv, making your project "importable" from anywhere within that environment
::  as if it were a third-party package you downloaded from the internet.
uv pip install -e .

:: Restore the original environment variables and exit
endlocal
