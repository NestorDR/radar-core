:: auto/context_mode_01_setup.cmd
:: Purpose: Install/update Context Mode (cuts context use when analyzing large files/repos)
:: Visit https://github.com/mksglu/context-mode/tree/main
:: Usage: auto/context_mode_01_setup.cmd (run from the project root directory)

@echo off

:: pnpm self-actualization
echo Installing/updating pnpm...
call pnpm self-update
if errorlevel 1 echo Warning: pnpm self-update failed; continuing with the installed version.

:: Context Mode actualization
echo Installing/updating Context Mode globally via pnpm...
call pnpm add -g context-mode


