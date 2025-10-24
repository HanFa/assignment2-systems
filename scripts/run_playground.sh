#!/usr/bin/env bash

uv run nsys profile --force-overwrite true -o trace/playground/nvtx_simple  python -m cs336_systems.playground.nvtx_simple
uv run nsys profile --force-overwrite true -o trace/playground/nvtx_mlp  python -m cs336_systems.playground.nvtx_mlp
uv run nsys profile --force-overwrite true -o trace/playground/nvtx_mlp_with_print  python -m cs336_systems.playground.nvtx_mlp_with_print

