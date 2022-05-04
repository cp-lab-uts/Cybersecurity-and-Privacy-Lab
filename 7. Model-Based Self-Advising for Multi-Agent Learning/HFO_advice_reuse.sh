#!/bin/bash

./bin/HFO --offense-agents=3 --defense-npcs=1 --trials 1000 --no-logging --headless &
sleep 5

# -x is needed to skip first line - otherwise whatever default python version is will run
python ./example/HFO_advice_reuse.py --port 6000 &> agent1.txt &
sleep 5
python ./example/HFO_advice_reuse.py --port 6000 &> agent2.txt &
sleep 5
python ./example/HFO_advice_reuse.py --port 6000 &> agent3.txt &

# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait
