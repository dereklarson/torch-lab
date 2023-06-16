#!/bin/bash
# Run this script as  `./update.sh torchlab` to rebuild the image

docker-compose -f docker/$1/docker-compose.yaml build
