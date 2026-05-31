#!/usr/bin/env bash 

set -e

for i in {1..10}; do 
	./overseer.sh 
	./no_overseer.sh
done
