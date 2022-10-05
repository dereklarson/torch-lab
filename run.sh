#!/bin/bash
# Run this script as  `./run.sh <app>` where app is:
#   "dev" to start the dev environment
#   "jupyterlab" to run the Jupyterlab server
#   "streamlit" to run the Streamlit UI

# For "dev" we want an interactive shell, so we use docker-compose run.
# In this case, we also need to add the service name "arc_dev"
# For the others, we want a standard attached container, via "up"
if [ "$1" == "dev" ]
then
   cmd="run arc_dev"
else
   cmd=up
fi
docker-compose -f docker/$1/docker-compose.yaml $cmd
