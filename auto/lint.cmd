:: auto/lint.cmd
:: Purpose: Linter and autoformat.

CLS

:: UVX: In most cases, executing a tool with `uvx` is more appropriate than installing the tool (https://docs.astral.sh/uv/concepts/tools/#execution-vs-installation)

:: Type checker and language server - ACTIVATE if you want the code to be stricter (good luck!)
:: uvx ty check .

:: Linter 
uvx ruff check . --diff
uvx ruff check . --fix

:: To request the latest version of Ruff and refresh the cache, use the @latest suffix:
:: uvx ruff@latest check . --fix

:: Very aggressive Formatter
::uvx ruff format
