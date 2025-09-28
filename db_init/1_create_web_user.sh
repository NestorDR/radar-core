#!/bin/bash
# Create a non-superuser 'web user' for the web apps
# This script runs before any .sql files to ensure the user exists before permissions are granted to it.

# Exit immediately if any command fails.
set -e

# Execute a SQL block to create the user and grant connect privileges.
# The user's password is provided by the $WEB_USER_PASSWORD environment variable.
# '-v ON_ERROR_STOP=1' tells psql to treat the entire SQL block as a single transaction
# '<<-EOSQL' & 'EOSQL' delimit the SQL instruction block
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER webuser WITH PASSWORD '$POSTGRES_WEBUSER_PASSWORD';
    GRANT CONNECT ON DATABASE "$POSTGRES_DB" TO webuser;
EOSQL