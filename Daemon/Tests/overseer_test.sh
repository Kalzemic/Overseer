#!/usr/bin/env bash 

set -e

for i in {1..2}; do 
	./overseer.sh 
	./no_overseer.sh
done
