#!/bin/bash
docker run --shm-size=24gb -it --rm -v /home/group5/:/workspace 3h4m/haste python3 -m ai_haste $@
