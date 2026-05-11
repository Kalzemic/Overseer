#!/usr/bin/env bash

docker rm -f no_overseer 2>/dev/null
docker run --name no_overseer --memory=0 stride-trainer
printf "\n"
docker logs -f no_overseer