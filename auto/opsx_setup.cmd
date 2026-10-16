:: auto\ospx_setup.cmd
:: Purpose:
::  1. Ensures latest @fission-ai/openspec CLI is installed globally.
::  2. Initializes OpenSpec in the current project repository.
:: Usage: auto\ospx_setup.cmd (run from the project root directory)

@echo off

:: pnpm self-actualization
echo Installing/updating pnpm...
call pnpm self-update
if errorlevel 1 echo Warning: pnpm self-update failed; continuing with the installed version.

:: OpenSpec actualization
echo Installing/updating OpenSpec CLI globally via pnpm...
call pnpm add -g @fission-ai/openspec@latest

if %errorlevel% neq 0 (
    echo [ERROR] Global installation of @fission-ai/openspec failed. Please check pnpm setup.
    exit /b %errorlevel%
)

echo.
echo Initializing OpenSpec in current project directory...
call openspec init

if %errorlevel% equ 0 (
    echo.
    echo ==============================================================================
    echo [SUCCESS] OpenSpec CLI installed globally and project initialized successfully!
    echo ==============================================================================
) else (
    echo.
    echo [ERROR] "openspec init" failed. Please check the terminal output above.
    exit /b %errorlevel%
)

:: Getting started:
::  Start your first change: /opsx-propose "your idea" (Antigravity, Junie)
::  Start your first change: /opsx:propose "your idea" (Claude Code)
::  Start your first change: $openspec-propose "your idea" (Codex CLI or IDE); in the Codex desktop app, select openspec-propose from Skills in the sidebar (Codex)