:: auto/once/clone_repo.cmd
:: Purpose: Clone the radar repositories and create the v0.6.x branch

@echo off
cls

:: setlocal preserves the original environment variables; endlocal restores them before exit
setlocal

set "REPOS_DIR=C:\Development\Repos\radar"

:: 1. Create the parent folder if it does not already exist
if not exist "%REPOS_DIR%" (
    mkdir "%REPOS_DIR%" || exit /b
)
cd /d "%REPOS_DIR%" || exit /b

:: 2. Clone the repositories
git clone https://github.com/NestorDR/radar-core.git "%REPOS_DIR%\radar-core" || exit /b
git clone https://github.com/NestorDR/radar-infra.git "%REPOS_DIR%\radar-infra" || exit /b

:: 3. Create and publish the v0.6.x branch
cd /d "%REPOS_DIR%\radar-core" || exit /b
git checkout -b v0.6.x || exit /b
git push -u origin v0.6.x || exit /b

:: Restore the original environment variables and exit
endlocal
