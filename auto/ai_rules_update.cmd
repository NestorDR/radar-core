:: auto/ai_rules_update.cmd
:: Purpose: Copy AI rules (as rtk) to the current project (.agents\rules, .claude\rules, .junie\rules)
::          Visit https://github.com/rtk-ai/rtk
:: Usage: auto/ai_rules_update.cmd (run from the project root directory)

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

set "AI_RULES_FOLDER=C:\Development\Learning\AI Harnessing\rules"
if not exist "%AI_RULES_FOLDER%" (
    echo Error: AI rules folder not found: %AI_RULES_FOLDER%
    set "_EXIT_CODE=1"
    goto end
)

:: Copy RTK rules.
echo Copying RTK rules...

:: If exist folder .agents\rules, copy
if exist .agents\rules (
    robocopy "%AI_RULES_FOLDER%" .agents\rules /E /COPY:DAT || set "_EXIT_CODE=1"
    if %_EXIT_CODE% neq 0 goto end
)

:: If exist folder .claude\rules, copy
if exist .claude\rules (
    robocopy "%AI_RULES_FOLDER%" .claude\rules /E /COPY:DAT || set "_EXIT_CODE=1"
    if %_EXIT_CODE% neq 0 goto end
)

:: If exist folder .junie\rules, copy
if exist .junie\rules (
    robocopy "%AI_RULES_FOLDER%" .junie\rules /E /COPY:DAT || set "_EXIT_CODE=1"
    if %_EXIT_CODE% neq 0 goto end
)

echo Done.

:end
:: Restore the original environment variables and exit with the resulting exit code
endlocal & exit /b %_EXIT_CODE%
