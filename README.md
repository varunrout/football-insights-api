# Football Insights Lab API

## Setup

1. Make sure your conda environment is activated  
2. Copy `.env.sample` to `.env` and adjust settings
3. Run: `uvicorn app.main:app --reload`
4. Visit http://localhost:8000/docs

## Environment Configuration

This project uses different environments for development and deployment:

- **TEST**: For testing with mock or limited data
- **DEV**: Development with real data
- **QUAL**: Pre-production validation
- **PROD**: Production deployment

Set the environment in your `.env` file:
```
ENVIRONMENT=DEV
```

# Football Insights API

This API provides tools for analyzing football (soccer) data from StatsBomb.

## FootballDataManager

The `FootballDataManager` class is a powerful utility for managing football data across multiple competitions with in-memory caching and serialization capabilities.

### Key Features

- Efficient in-memory caching of competitions, matches, events, and 360° data
- Serialization and deserialization of data to/from disk
- Comprehensive dataset preparation for analysis
- Support for StatsBomb's open data

### Installation

```python
# Dependencies
pip install pandas statsbombpy
```

### Usage Examples

#### Basic Data Retrieval

```python
from app.util.football_data_manager import FootballDataManager

# Initialize the manager
fdm = FootballDataManager(cache_dir="data_cache")

# Get available competitions (defaults to only competitions with 360 data and excluding women's competitions)
competitions = fdm.get_competitions()

# Get matches for a specific competition and season
matches = fdm.get_matches(competition_id=43, season_id=3)

# Get events for a specific match
events = fdm.get_events(match_id=3788741)

# Get 360° freeze frame data for a match (returns empty DataFrame if not available)
frames = fdm.get_freeze_frames(match_id=3788741)
```

#### Preparing Analysis Datasets

```python
# Prepare data for analysis (will fetch and process data for competitions)
analysis_data = fdm.prepare_data_for_analysis(
    competition_ids=[43, 2],  # Optional list of competition IDs to include
    max_matches_per_competition=5  # Limit matches per competition
)

# Save the analysis dataset to disk
dataset_path = fdm.save_analysis_dataset(
    analysis_data=analysis_data,
    dataset_name="premier_league_analysis"  # Optional name (defaults to timestamp)
)

# Later, load the saved dataset
loaded_data = fdm.load_analysis_dataset(
    dataset_path=dataset_path,
    load_data=True  # Set to False for metadata only (less memory usage)
)
```

### API Reference

#### Initialization

```python
fdm = FootballDataManager(cache_dir="data_cache")
```

#### Competition Data

- `get_competitions(force_refresh=False, only_with_360=True, exclude_women=True)`: Retrieves available competitions with optional filtering.

#### Match Data

- `get_matches(competition_id, season_id, force_refresh=False)`: Retrieves matches for a specific competition and season.

#### Event Data

- `get_events(match_id, force_refresh=False)`: Retrieves event data for a specific match.
- `get_freeze_frames(match_id, force_refresh=False)`: Retrieves 360° freeze frame data for a specific match.

#### Data Serialization

- `serialize_data(data_type, identifier, data)`: Serializes data to disk.
- `load_serialized_data(data_type, identifier)`: Loads serialized data from disk.

#### Analysis Datasets

- `prepare_data_for_analysis(competition_ids=None, max_matches_per_competition=5)`: Prepares a comprehensive dataset for analysis.
- `save_analysis_dataset(analysis_data, dataset_name=None)`: Saves an analysis dataset to disk.
- `load_analysis_dataset(dataset_path, load_data=True)`: Loads a previously saved analysis dataset.

### Data Structure

The prepared analysis data has the following structure:

```
{
    'competitions': {
        competition_id: {
            'name': competition_name,
            'season': season_name,
            'matches': {
                match_id: {
                    'home_team': home_team_name,
                    'away_team': away_team_name,
                    'score': 'home_score-away_score',
                    'events': events_dataframe,
                    'freeze_frames': frames_dataframe,
                    'has_360_data': boolean
                }
            }
        }
    },
    'summary': {
        'total_competitions': int,
        'total_matches': int,
        'total_events': int,
        'matches_with_360': int
    }
}
```

## Notebooks

The repository contains Jupyter notebooks demonstrating different aspects of football data analysis.

### 01_data_preparation.ipynb

This notebook demonstrates the usage of the `FootballDataManager` to prepare football data for analysis:

1. **Initialization**: Setting up the data manager with a specified cache directory.
   ```python
   fdm = FootballDataManager()
   ```

2. **Retrieving Competitions**: Getting and filtering competitions with 360° data.
   ```python
   competitions = fdm.get_competitions(only_with_360=True, exclude_women=True)
   ```

3. **Data Preparation**: Creating a comprehensive dataset for analysis by selecting top competitions.
   ```python
   analysis_data = fdm.prepare_data_for_analysis(
       competition_ids=competition_ids,
       max_matches_per_competition=3
   )
   ```

4. **Sample Analysis**: Calculating average shots per match for each competition.
   ```python
   for comp_id, comp_data in analysis_data['competitions'].items():
       comp_name = comp_data['name']
       shots_per_match = []
       for match_id, match_data in comp_data['matches'].items():
           events_df = match_data['events']
           shots = events_df[events_df['type'] == 'Shot'].shape[0]
           shots_per_match.append(shots)
       avg_shots = sum(shots_per_match) / len(shots_per_match) if shots_per_match else 0
   ```

5. **Data Persistence**: Saving and loading analysis datasets.
   ```python
   # Save dataset
   dataset_path = fdm.save_analysis_dataset(analysis_data, dataset_name="top_10_competitions")
   
   # Load dataset (metadata only)
   loaded_data_metadata = fdm.load_analysis_dataset(dataset_path, load_data=False)
   
   # Load complete dataset
   loaded_data_complete = fdm.load_analysis_dataset(dataset_path, load_data=True)
   ```

The notebook demonstrates a complete workflow from data acquisition to analysis, showing how to efficiently manage football data across multiple competitions.

### 02_data_interpretation.ipynb

This notebook explores and analyzes the football data prepared in the previous notebook, focusing on understanding patterns and generating insights:

1. **Loading the Dataset**: Loading the preprocessed dataset from the cache.
   ```python
   fdm = FootballDataManager()
   dataset_path = "data_cache/top_10_competitions"
   analysis_data = fdm.load_analysis_dataset(dataset_path, load_data=True)
   ```

2. **Understanding Data Structure**: Examining the structure and types of data.
   ```python
   # Create a dataframe of competitions for easier analysis
   competitions_data = []
   for comp_id, comp_data in analysis_data['competitions'].items():
       comp_info = {
           'competition_id': comp_id,
           'name': comp_data['name'],
           'season': comp_data['season'],
           'num_matches': len(comp_data['matches']),
           'matches_with_360': sum(1 for m in comp_data['matches'].values() if m['has_360_data'])
       }
       competitions_data.append(comp_info)
   ```

3. **Event Type Analysis**: Analyzing the distribution of event types across matches.
   ```python
   # Collect event types across all matches
   all_event_types = []
   for comp_id, comp_data in analysis_data['competitions'].items():
       for match_id, match_data in comp_data['matches'].items():
           if 'events' in match_data:
               event_types = match_data['events']['type'].value_counts().to_dict()
               for event_type, count in event_types.items():
                   all_event_types.extend([event_type] * count)
   ```

4. **Shot Analysis**: Examining shot data, including outcomes, positions, and expected goals.
   ```python
   # Shot outcomes
   shot_outcomes = shots_df['shot_outcome'].value_counts()
   
   # Average xG by competition
   avg_xg = shots_df.groupby('competition')['shot_statsbomb_xg'].mean().sort_values(ascending=False)
   ```

5. **Pass Analysis**: Analyzing passing patterns, including success rates by competition.
   ```python
   # Calculate pass success rate
   if total_passes > 0:
       pass_success_by_comp[comp_name] = (successful_passes / total_passes) * 100
   ```

6. **Visualizations**: Creating football-specific visualizations like shot maps and player positioning.
   ```python
   def draw_pitch(ax):
       """Draw a football pitch on the given axes"""
       # Pitch dimensions in StatsBomb data: 120x80
       pitch_length = 120
       pitch_width = 80
       
       # Draw pitch elements
       # ...
   
   def create_shot_map(match_data):
       """Create a shot map for a specific match"""
       # Plot shots on the pitch with size representing xG
       # ...
   
   def visualize_freeze_frame(match_data, frame_id=None):
       """Visualize a single freeze frame from the 360 data"""
       # Plot player positions from a freeze frame
       # ...
   ```

7. **Key Findings**: Summarizing insights and identifying patterns in the data.
   - Event distribution patterns across competitions
   - Shot quality and efficiency variations
   - Pass success rates and team playing styles
   - Player performance metrics
   - Applications of 360° data for tactical analysis

The notebook provides a comprehensive analysis of football data across multiple competitions, demonstrating how to extract meaningful insights from complex event and tracking data.

### 03_metric_engineering.ipynb

This notebook focuses on developing advanced football metrics from event data, creating sophisticated performance indicators for teams and players:

1. **Building an Expected Threat (xT) Model**: Implementing a grid-based model that values pitch locations based on their goal probability.
   ```python
   # Create an xT model
   xt_model = ExpectedThreatModel(n_grid_cells_x=12, n_grid_cells_y=8)
   xt_grid = xt_model.train(full_events_df)
   
   # Visualize the xT model
   xt_model.visualize_grid(ax=ax, title="Expected Threat (xT) Model")
   ```

2. **Basic Performance Metrics**: Calculating foundational metrics for teams.
   ```python
   # Calculate metrics for a match
   metrics = calculate_basic_match_metrics(events_df)
   
   # Basic metrics include:
   # - Possession percentage
   # - Pass completion rate
   # - Shots and xG
   # - Defensive actions
   ```

3. **Advanced Metric Implementation**:
   - **xT Added by Actions**: Measuring the value added by passes and carries.
     ```python
     events_with_xt = calculate_xt_added(events_df, xt_model)
     ```
   
   - **PPDA (Passes Per Defensive Action)**: Quantifying pressing intensity.
     ```python
     ppda = calculate_ppda(events_df, team, opposition_half_only=True)
     ```
   
   - **Progressive Passes**: Identifying passes that significantly advance the ball.
     ```python
     progressive_passes = identify_progressive_passes(events_df, distance_threshold=10)
     ```

4. **Player Performance Analysis**: Evaluating player contributions using the new metrics.
   ```python
   # Top players by total xT added
   top_players_xt = player_xt_df.sort_values('total_xt_added', ascending=False)
   
   # Top players by average xT per action
   top_players_avg_xt = player_xt_df.sort_values('avg_xt_per_action', ascending=False)
   ```

5. **Exporting Metrics for API**: Saving models and calculated metrics for use in the API.
   ```python
   # Export xT model
   with open(export_dir / 'xt_model.pkl', 'wb') as f:
       pickle.dump(xt_model, f)
   ```

The notebook demonstrates how to transform raw event data into meaningful football metrics that provide deeper tactical insights and player evaluation tools. These metrics form the foundation for more advanced analysis and can be integrated into the API for frontend visualization and reporting.

## Project Structure

```
/workspaces/football-insights-api/
├── app/                      # Application code (PROD)
│   ├── config/               # Configuration
│   │   └── environment.py    # Environment settings
│   ├── util/                 # Utilities and helpers
│   │   └── football_data_manager.py
│   ├── api/                  # API endpoints
│   └── main.py               # FastAPI application
├── notebooks/                # Jupyter notebooks (TEST/DEV)
│   ├── 01_data_preparation.ipynb
│   ├── 02_data_interpretation.ipynb
│   └── 03_metric_engineering.ipynb
├── data_cache/               # Cached data
├── .env                      # Environment variables
└── README.md                 # Documentation
```

## Development Workflow

1. Develop and test analysis in Jupyter notebooks (TEST/DEV)
2. Create and test API endpoints in notebooks (DEV)
3. Migrate validated code to Python modules (QUAL/PROD)
4. Deploy to production environment

See [plan.md](plan.md) for detailed project roadmap and visualization plan.
