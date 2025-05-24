# Football Insights API Project Plan

## Project Overview

This project aims to build a comprehensive football analytics platform leveraging StatsBomb data to provide advanced insights through:
1. Data preparation and caching
2. Advanced metric engineering (xT, PPDA, etc.)
3. Visualization dashboards 
4. API endpoints for frontend consumption

## Environment Setup

We're using multiple environments for systematic development:

1. **TEST**
   - Purpose: Initial testing of functions and analyses
   - Data: Mock data or limited real data samples
   - Tools: Jupyter notebooks, unit tests
   - Location: `/notebooks`, `/tests`

2. **DEV**
   - Purpose: Development with real data
   - Data: Complete StatsBomb dataset
   - Tools: Jupyter notebooks, local API server
   - Location: `/notebooks`, `/app` (for API development)

3. **QUAL**
   - Purpose: Pre-production validation
   - Data: Mirror of production data
   - Tools: Python modules only (no notebooks)
   - Location: `/app`

4. **PROD**
   - Purpose: Production deployment
   - Data: Complete StatsBomb dataset
   - Tools: Python modules only
   - Location: `/app`

## Development Workflow

Our development follows this pattern:
1. Develop and test analysis in Jupyter notebooks (TEST/DEV)
2. Create and test API endpoints in notebooks (DEV)
3. Migrate validated code to Python modules (QUAL/PROD)
4. Deploy to production environment

## Current Progress

1. ✅ Set up `FootballDataManager` for efficient data handling
2. ✅ Created notebooks for data preparation
3. ✅ Created notebooks for data interpretation
4. ✅ Created notebooks for metric engineering
5. 🔄 Working on API endpoint development

## Next Steps

1. Define API endpoints for each visualization category
2. Create endpoint testing notebooks
3. Implement visualization-specific calculations
4. Migrate core functions to Python modules
5. Set up FastAPI endpoints

## Visualization Plan

We'll develop dashboards in these main categories:

### 1. Dashboard (KPI Summary)
- Season KPI cards with trends
- xG vs Goals over time
- Defensive & possession metrics
- Shot zone and pass network visualizations

### 2. xT Analytics
- xT pitch heatmap
- Player xT rankings
- xT passing map
- Cumulative xT timeline

### 3. Player Comparison
- Player radar charts
- Dual bar comparisons
- Style similarity maps
- Role comparison tables

### 4. Tactical Insights
- Offensive analysis (pass networks, shot creation)
- Defensive analysis (action maps, PPDA)
- Build-up analysis (possession progression)
- Transition analysis
- Set-piece analysis

### 5. Positional Analysis
- Role analysis
- Zone analysis

### 6. Match-Up Analysis
- Opposition analysis
- League benchmarks

### 7. Player Analysis
- Player profiles
- Performance metrics
- Heatmaps and action maps
- Trend analysis

## Implementation Approach

1. **Data Processing Layer**
   - Core data retrieval and caching (FootballDataManager)
   - Metric calculation modules
   - Data transformation utilities

2. **API Layer**
   - FastAPI endpoints for each visualization
   - Parameter validation
   - Response formatting

3. **Visualization Layer** (Frontend)
   - Next.js application
   - Chart components using libraries like D3, Plotly, or Chart.js
   - Dashboard layouts and filters

## Technical Components

1. **Data Processing**
   - `FootballDataManager` for caching and retrieval
   - Metric calculation classes (xT, PPDA, etc.)
   - Event processing utilities

2. **API Structure**
   - Main routers:
     - `/api/dashboard`
     - `/api/xt`
     - `/api/player`
     - `/api/tactics`
     - `/api/positions`
     - `/api/matchups`
     - `/api/players`

3. **Frontend Integration**
   - API client for data fetching
   - Visualization components
   - Interactive dashboard layouts

## Timeline

1. **Phase 1: Core Data & Metrics** (Current)
   - Complete data preparation
   - Implement all advanced metrics
   - Set up testing framework

2. **Phase 2: API Development**
   - Design API endpoints
   - Implement and test all endpoints
   - Document API interfaces

3. **Phase 3: Frontend Integration**
   - Connect API endpoints
   - Build visualization components
   - Create interactive dashboards

4. **Phase 4: Deployment & Optimization**
   - Deploy to production
   - Optimize performance
   - Add additional features

## Notebook Organization

```
/notebooks
├── 01_data_preparation.ipynb
├── 02_data_interpretation.ipynb
├── 03_metric_engineering.ipynb
├── api_tests/
│   ├── dashboard_api_test.ipynb
│   ├── player_api_test.ipynb
│   └── ...
└── visualizations/
    ├── dashboard_viz.ipynb
    ├── player_viz.ipynb
    └── ...
```

## Python Module Organization

```
/app
├── util/
│   ├── football_data_manager.py
│   ├── metrics/
│   │   ├── expected_threat.py
│   │   ├── ppda.py
│   │   └── ...
│   └── viz/
│       ├── pitch.py
│       ├── charts.py
│       └── ...
├── api/
│   ├── dashboard.py
│   ├── player.py
│   └── ...
└── main.py
```
