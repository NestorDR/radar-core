:: auto/ponytail_update.cmd
:: Purpose: Copy Ponytail skills to the current project  (.agents\rules, .claude\rules, .junie\rules)
::          Visit https://github.com/DietrichGebert/ponytail/tree/main
::                https://pi.dev/packages/@dietrichgebert/ponytail
:: Usage: auto/ponytail_update.cmd (run from the project root directory)

@echo off

:: setlocal preserves the original environment variables; endlocal restores them before exit
setlocal
set "_EXIT_CODE=0"

:: Require invocation from this repository's root directory
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
for %%I in ("%CD%") do set "CURRENT_DIR=%%~fI"
if /i not "%CURRENT_DIR%"=="%PROJECT_ROOT%" (
    echo Error: Run this script from the project root: "%PROJECT_ROOT%"
    set "_EXIT_CODE=1"
    goto end
)

set "PONYTAIL_FOLDER=C:\Development\Repos\ponytail"

:: First time: get temporary ponytail repo
:: git clone --depth 1 https://github.com/DietrichGebert/ponytail.git %PONYTAIL_FOLDER%
:: if %_EXIT_CODE% neq 0 goto end

:: Update ponytail with latest source code.
git -C "%PONYTAIL_FOLDER%" pull origin main || set "_EXIT_CODE=1"
if %_EXIT_CODE% neq 0 goto end

:: Copy ponytail skills.
echo Copying ponytail skills...

:: If exist folder .agents\skills, copy
if exist .agents\skills (
    robocopy "%PONYTAIL_FOLDER%\skills\ponytail" .agents\skills\ponytail /E /COPY:DAT || set "_EXIT_CODE=1"
    if %_EXIT_CODE% neq 0 goto end
)

:: If exist folder .claude\skills, copy
if exist .claude\skills (
    robocopy "%PONYTAIL_FOLDER%\skills\ponytail" .claude\skills\ponytail /E /COPY:DAT || set "_EXIT_CODE=1"
    if %_EXIT_CODE% neq 0 goto end
)

:: If exist folder .junie\skills, copy
if exist .junie\skills (
    robocopy "%PONYTAIL_FOLDER%\skills\ponytail" .junie\skills\ponytail /E /COPY:DAT || set "_EXIT_CODE=1"
    if %_EXIT_CODE% neq 0 goto end
)

:: Remove ponytail repo
:: rmdir /s /q %PONYTAIL_FOLDER%
:: if %_EXIT_CODE% neq 0 goto end

echo Done.

:end
:: Restore the original environment variables and exit with the resulting exit code
endlocal & exit /b %_EXIT_CODE%

:: Visit https://pi.dev/packages/@dietrichgebert/ponytail
:: How to Verify and Control It
:: /ponytail full       → Supported levels: lite, full [default], ultra, or off
:: /ponytail-review     → Checks bugs, security, real load, missing tests, speed, and what to cut
:: /ponytail-audit      → Ranks what it finds and tells you what to fix first
