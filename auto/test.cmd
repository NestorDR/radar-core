:: auto/test.cmd
:: Purpose: Run all tests in the project and clean up temporary test artifacts

@echo off

:: 1. Set the virtual environment for this test run.
:: setlocal preserves the original VIRTUAL_ENV; endlocal restores it before exit.
set "VIRTUAL_ENV=C:\Development\VirtualEnvs\radar-3.13.14-uv-env"

:: 2. Run pytest with --cache-clear and route temporary fixture files to system TEMP
uv run --active pytest %1 --cache-clear --basetemp="%TEMP%\pytest_runner"
set TEST_EXIT_CODE=%ERRORLEVEL%

:: 3. Post-test cleanup of project root artifacts
echo Cleaning up temporary test artifacts...

:: Remove .pytest_cache if created locally
if exist .pytest_cache rmdir /s /q .pytest_cache

:: Remove Python bytecode directories (__pycache__) recursively
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

:: Remove coverage reports (if using pytest-cov)
if exist .coverage del /f /q .coverage
if exist htmlcov rmdir /s /q htmlcov
if exist .coverage.* del /f /q .coverage.*

:: 4. Restore the original VIRTUAL_ENV and exit with the original pytest exit code.
endlocal & exit /b %TEST_EXIT_CODE%