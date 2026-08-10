#!/usr/bin/env bash
# auto/ci_test_install_talib.sh
# Purpose: Pre-requisite setup for STAGE 1: TESTING in GitHub Actions CI/CD pipeline (.github/workflows/ci-cd-pipeline.yml).
# Installs the underlying TA-Lib C shared library (/usr/lib/libta-lib.so) on Linux runner environments so that
# Pytest can execute technical indicator tests (RSI, ATR) without shared object load errors.

set -euo pipefail

# Download the TA-Lib C source code archive
wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz

# Extract the tarball archive
tar -xzf ta-lib-0.6.4-src.tar.gz

# Navigate into the extracted source code directory
cd ta-lib-0.6.4/

# Configure the Makefile for system installation path /usr
./configure --prefix=/usr

# Compile C source code into shared object library files (.so)
make

# Install compiled library files into system directories (/usr/lib)
if [ "$(id -u)" -eq 0 ]; then
  make install
else
  sudo make install
fi

# Return to root workspace directory and clean up temporary files
cd ..
rm -rf ta-lib*
