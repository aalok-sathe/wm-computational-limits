#!/usr/bin/env python

import os
import sys
import subprocess

venv_python = os.path.join(".venv", "bin", "python")

# Process arguments to split multi-value arguments like "--dataset.concurrent_reg=2 3 4"
processed_args = []
for arg in sys.argv[1:]:
    processed_args += arg.replace('"', "").split(" ")

new_cmd = [venv_python, "-m", "workingmem", *processed_args]
print(new_cmd)

subprocess.run(new_cmd)
