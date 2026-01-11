# docker/Dockerfile

# Stage 1 Builder: contains build tools to build the radar_core wheel
# The --platform=$BUILDPLATFORM ensures this stage runs on your host's native architecture.
FROM --platform=$BUILDPLATFORM python:3.13-slim-bookworm AS builder

# Copy the 'uv' binary for the specific version from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install build-time dependencies for TA-Lib, psycopg2, and other tools.
# - apt-get update: Refreshes package index with latest versions
# - apt-get install
#   -y: automatic yes to prompts (non-interactive installation)
#   --no-install-recommends: install only essential dependencies
# - Packages:
#   wget: Utility to download files from the internet (for TA-Lib source).
#   build-essential: Provides compilers and libraries for building Python C extensions (includes 'make' utility)
#   gcc: GNU Compiler Collection (C compiler) needed to compile C code (TA-Lib and Python C-extensions)
#   libpq-dev: Development headers for PostgreSQL, needed to compile psycopg2
# - rm -rf /var/lib/apt/lists/*: removes package lists downloaded with `apt-get update`
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        gcc \
        libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Check for TA-Lib updates: https://github.com/ta-lib/ta-lib/releases
# Download, compile, and install the TA-Lib C library from its source code.
# - wget http://...: Download the TA-Lib source code archive
# - tar -xzf ...: Extract the archive.
# - cd ta-lib/: Change to extracted directory with the source code
# - ./configure --prefix=/usr: Configure the build script for the current system environment
# - make: Compile the source code using the Makefile created by './configure' (creates the shared library 'libta_lib.so.0.0.0')
# - make install: Install the compiled library into the system directories
# - cd ..: Go back to the parent directory, and
# - rm -rf ta-lib*: Clean up downloaded/extracted files

RUN wget https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar -xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# Set env variables to tell the Python TA-Lib wrapper's build script where to find the C library's headers and library files.
ENV TA_LIBRARY_PATH=/usr/lib
ENV TA_INCLUDE_PATH=/usr/include

# Set the working directory for the application build.
WORKDIR /app

# Prepare wheels directory early
RUN mkdir -p /wheels

# Copy pyproject.toml (the only true source, not use requirements.txt)
COPY ../pyproject.toml .

# Compile (download & package) deps as wheels in a single atomic layer to optimize caching
# (this layer will only be rebuilt if pyproject.toml changes).
# 1. uv compile third-party deps into a requirements.txt file without attempting to build the local project.
# 2. pip Prebuild and cache all dependencies as wheels in the /wheels directory to enable faster,
#    offline-capable installations later in the build or runtime.
RUN uv pip compile pyproject.toml -o requirements.txt && \
    pip wheel --wheel-dir=/wheels -r requirements.txt

# Copy files needed to build the wheel (radar_core source code and Docker-specific settings.)
COPY ./src ./src
COPY ./README.md ./README.md

# Build the wheel for local application.
RUN uv build --wheel --out-dir /wheels




# Stage 2 Final App: lean production image.
# It uses the target architecture specified during the build (e.g., linux/arm64).
FROM python:3.13-slim-bookworm AS final

# Metadata of the Docker image
LABEL maintainer="ndr1970@gmail.com"
LABEL version="0.2.0"

# Environment variables for better Python behavior in containers
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Numba Cache: Ensure it writes to a user-writable directory
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Install only runtime dependencies
# - libpq5: The PostgreSQL client library needed by psycopg2 to connect
# - libgomp1: GNU OpenMP required for Numba parallel execution-support
# - curl: A useful utility for health checks (or debugging)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libpq5 \
        curl &&  \
    apt-get clean &&  \
    rm -rf /var/lib/apt/lists/*

# Manually copy the compiled TA-Lib shared library from the builder stage
COPY --from=builder /usr/lib/libta-lib.so.0 /usr/lib/
COPY --from=builder /usr/lib/libta-lib.so.0.0.0 /usr/lib/

# Update the linker cache to make the newly added library discoverable by the system.
RUN ldconfig

# Copy the built wheel from the 'builder' stage
COPY --from=builder /wheels /wheels

# Use standard pip to install the app wheel and its dependencies (to keep the image minimal) from the local directory, then cleanup.
# --no-index ensures pip does NOT access any network repository.
RUN pip install --no-cache-dir --no-index /wheels/*.whl && rm -rf /wheels

# Create the non-root user for security and set its home directory as the working directory.
RUN useradd --uid 1001 --create-home default
WORKDIR /home/default/app

# Copy settings file and assign the 'default' user property
COPY --chown=default:default ./src/radar_core/settings.yml ./settings.yml

# Switch to the non-root user for runtime execution.
USER default

# Run entry point.
CMD ["python", "-m", "radar_core"]
