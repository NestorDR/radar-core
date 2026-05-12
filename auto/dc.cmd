:: auto/dc.cmd
:: Purpose: Start  the Docker Compose environment for a specific deployment environment: core, mb
:: Usage: auto/dc <target_deploy>
@ECHO OFF
CLS

:: Enable `delayed expansion` to handle environment variables correctly within blocks
SETLOCAL ENABLEDELAYEDEXPANSION
ECHO !DATE! !TIME!

:: Get the directory where the script is located
SET SCRIPT_DIR=%~dp0
:: Ensure we are executing from the project root (one level up from /auto)
CD /D "!SCRIPT_DIR!..!"

:: Capture 1st argument
SET TARGET_DEPLOY=%1
if "!TARGET_DEPLOY!"=="" (
    ECHO Parameter is required. Usage: %0 ^<target_deploy^>
    ECHO Valid values for target_deploy: core, mb
    ECHO Example: %0 core
    exit /b
)

:: Since `docker compose` is case-sensitive, force `TARGET_DEPLOY` to lowercase to match the
:: `docker-compose.*.yml` and `.env.*` filenames, ensuring consistent resolution regardless of user input casing.
:: /i : Case-insensitive comparison
if /i "!TARGET_DEPLOY!"=="core" (
    SET TARGET_DEPLOY=
    SET TARGET_DEPLOY=core
)
if /i "!TARGET_DEPLOY!"=="mb" (
    SET TARGET_DEPLOY=
    SET TARGET_DEPLOY=mb
)

:compose
:: DOCKER-COMPOSE
:: Visit: https://docs.docker.com/reference/cli/docker/compose/up/

:: FLAGS EXPLANATION:
:: Only Build
:: build ..........: builds images before starting containers
:: --no-cache .....: to force rebuild
:: ^  : newline
:: && : starts next command if the previous one ended successfully
:: ```cmd
:: docker compose -f docker/docker-compose.!TARGET_DEPLOY!.yml build --no-cache ^
:: && docker compose ... up
:: ```

:: Build and launch all services as detached
:: --env-file .....: to inject the environment variables from a specific file
:: -f .............: to specify a particular compose file .yml
:: -p .............: to add an isolated specific project name
:: up .............: to raise the services defined in the .yml file
:: -d .............: to run in background (detached mode), visible in Docker Desktop
:: --build ........: reconstructs images if it detects changes in the build context or the Dockerfile
:: --force-recreate: ensures that containers are created again (stops and removes existing containers and creates new ones)
:: %* .............: for extra arguments
docker compose --env-file envs/.env.!TARGET_DEPLOY! -f docker/docker-compose.!TARGET_DEPLOY!.yml -p radar-!TARGET_DEPLOY! up -d --build

:: Check if the command was successful
IF !errorlevel! neq 0 (
    ECHO [ERROR] Docker Compose for !TARGET_DEPLOY! Environment failed to start.
    EXIT /b !errorlevel!
)

ECHO [SUCCESS] !TARGET_DEPLOY! Environment is up.
ECHO !DATE! !TIME!

ENDLOCAL
