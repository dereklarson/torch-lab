#!/bin/bash
# Run this script as  `./update.sh <app>` to rebuild the app:
#   "dev", "jupyterlab", "streamlit"

docker-compose -f docker/$1/docker-compose.yaml build
