#!/bin/bash

# Define folder paths
folders=("routers" "services" "schemas" "metrics")

# Define files to be created under each relevant folder
files=(
  "tactical_offensive"
  "tactical_defensive"
  "tactical_buildup"
  "tactical_transitions"
  "tactical_setpiece"
  "positional_role"
  "positional_zone"
  "matchup_opposition"
  "matchup_benchmark"
  "player_profile"
  "player_metrics"
  "player_heatmaps"
  "player_trends"
  "player_percentile"
  "dashboard_main"
  "xt_main"
)

# Create folders and __init__.py
for folder in "${folders[@]}"; do
  mkdir -p "$folder"
  touch "$folder/__init__.py"
done

# Create Python module files in routers/, services/, schemas/
for file in "${files[@]}"; do
  touch "routers/${file}.py"
  touch "services/${file}.py"
  touch "schemas/${file}.py"
done

# Optional: Create placeholder metric files
touch metrics/xt_model.py
touch metrics/xg_model.py

echo "✅ All files and folders created."
