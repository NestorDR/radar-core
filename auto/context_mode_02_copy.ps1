# auto/context_mode_02_copy.ps1
# Purpose: copy Context Mode instructions for Antigravity (no hook support)
# Usage: auto/context_mode_02_copy.ps1 (run from the project root directory)

Invoke-WebRequest -Uri "https://raw.githubusercontent.com/mksglu/context-mode/main/configs/antigravity/GEMINI.md" -OutFile ".\GEMINI.md"