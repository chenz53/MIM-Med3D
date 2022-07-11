#!/usr/bin/env bash

MAIN_FILE=$1
CONFIG_FILE=$2
python3 $MAIN_FILE fit --config $CONFIG_FILE
