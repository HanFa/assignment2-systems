#!/usr/bin/env bash

uv run nsys profile --force-overwrite true -o trace/benchmark_attn  python -m cs336_systems.benchmark_attn
