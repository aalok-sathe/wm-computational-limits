#!/usr/bin/env python

import os
import sys
import subprocess

venv_python = os.path.join(".venv", "bin", "python")
subprocess.run([venv_python, "-m", "workingmem", *sys.argv[1:]])
