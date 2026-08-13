:: auto/test.cmd
:: Purpose: Run all tests in the project and clean up temporary test artifacts

@echo off

:: 1. Run pytest with --cache-clear and route temporary fixture files to system TEMP
uv run --active pytest %1 --cache-clear --basetemp="%TEMP%\pytest_runner"
set TEST_EXIT_CODE=%ERRORLEVEL%

:: 2. Post-test cleanup of project root artifacts
echo Cleaning up temporary test artifacts...

:: Remove .pytest_cache if created locally
if exist .pytest_cache rmdir /s /q .pytest_cache

:: Remove Python bytecode directories (__pycache__) recursively
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

:: Remove coverage reports (if using pytest-cov)
if exist .coverage del /f /q .coverage
if exist htmlcov rmdir /s /q htmlcov
if exist .coverage.* del /f /q .coverage.*

:: 3. Exit with the original pytest exit code so CI/CD pipelines detect test failures
exit /b %TEST_EXIT_CODE%