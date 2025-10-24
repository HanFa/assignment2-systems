#!/usr/bin/env bash

uv run nsys profile --force-overwrite true -o trace/nsys_profile  python -m cs336_systems.nsys_profile
