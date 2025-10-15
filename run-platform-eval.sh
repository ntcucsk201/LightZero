#!/bin/bash

SESSION="darkchess"

tmux new-session -d -s $SESSION

tmux send-keys -t $SESSION:0 'java -jar platform/open/Launcher.jar' C-m
tmux send-keys -t $SESSION:0 '6' C-m

tmux split-window -h -p 50 -t $SESSION:0
tmux send-keys -t $SESSION:0 'java -jar platform/enter/Launcher.jar' C-m
# tmux send-keys -t $SESSION:0 C-m
# tmux send-keys -t $SESSION:0 '6' C-m
# tmux send-keys -t $SESSION:0 '1' C-m

tmux attach -t $SESSION
