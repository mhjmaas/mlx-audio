#!/bin/bash

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Start the MLX Audio server with UI
python -m mlx_audio.server --host 0.0.0.0 --start-ui --workers 1 "$@"
