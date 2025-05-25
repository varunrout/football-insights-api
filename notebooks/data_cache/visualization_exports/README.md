# Football API Visualization Examples

This directory contains example visualizations that can be returned by the Football Insights API endpoints.
Each JSON file contains sample API responses with base64-encoded visualizations that can be displayed in a web application.

## Available Visualizations

1. **Expected Threat (xT) Analytics**
   - xT Model Grid Visualization
   - xT Pass Map Visualization

2. **Player Comparison**
   - Radar Chart for Player Comparison
   - Scatter Plot for Player Comparison

3. **Matchup Analysis**
   - Head-to-Head Analysis
   - Team Style Comparison

4. **Positional Analysis**
   - Zone Effectiveness Analysis
   - Player Heat Map

## Usage

These visualizations can be directly included in web applications by embedding the base64-encoded images:

```html
<img src="data:image/png;base64,{visualization_base64}" alt="Visualization">
```

Or they can be used as templates to create similar visualizations for the actual API responses.
