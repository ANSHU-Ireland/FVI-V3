# FVI Phase 1 - Fossil Vulnerability Index

A comprehensive assessment system for evaluating fossil fuel vulnerability across five target countries: India (IND), China (CHN), United States (USA), Japan (JPN), and South Africa (ZAF).

## Quick Start

1. **Setup the environment:**
   ```bash
   python setup.py
   ```

2. **Run the assessment:**
   ```bash
   python main.py
   ```

3. **Follow the interactive prompts** to provide your profile and get a personalized assessment.

## Project Structure

```
FVI/
├── main.py                 # Main application entry point
├── setup.py               # Setup and validation script
├── requirements.txt       # Python dependencies
├── config/
│   └── config.yaml        # Configuration settings
├── src/
│   ├── data_processor.py  # Data ingestion and cleaning
│   ├── metrics_calculator.py # Pillar metric calculations
│   ├── weight_engine.py   # Dynamic weighting system
│   ├── classifier.py      # Classification logic
│   ├── facts_generator.py # Gemini explanation engine
│   └── validator.py       # Validation and QA
├── data/
│   ├── raw/              # Raw data files (auto-populated)
│   ├── clean/            # Cleaned canonical tables
│   └── templates/        # Micro-tables for manual input
├── artifacts/
│   ├── scores/           # Country scores and classifications
│   ├── facts/            # LLM facts for explanations
│   └── logs/             # Run metadata and QA reports
└── docs/
    ├── SCHEMA.md          # Data schema specification
    └── RUNBOOK_PHASE1.md  # Implementation runbook
```

## Features

### Data Processing
- Automated ingestion from `../data-sources` folder
- Deterministic reference year selection
- Data quality scoring and validation
- Country filtering (restricted to 5 target countries)

### Pillar Metrics
- **Emissions**: Absolute emissions, intensity, coverage gaps
- **Necessity**: Coal dependency, consumption lock-in, population at risk
- **Ecological**: Land disturbance, site density, deforestation exposure
- **Artificial Support**: Exit penalties, state ownership, coverage gaps

### Dynamic Weighting
- Persona-driven adjustments (investor/regulator/operator)
- Time horizon considerations (0-5y/5-10y/10-30y)
- Risk tolerance and capital capacity factors
- Depth-aware scaling (quick/standard/deep analysis)

### Classification System
- Three categories: Sustainable / Critical Transition / Decommission
- Guardrail rules for policy constraints
- Borderline detection and flagging

### AI-Powered Explanations
- Gemini-generated explanations grounded in facts
- Personalized recommendations based on user profile
- Clarifying questions for data gaps

### Quality Assurance
- Comprehensive validation gates
- Cross-check inconsistency detection
- Provenance tracking for all metrics
- Detailed QA reports

## Configuration

Edit `config/config.yaml` to customize:
- Reference year policy
- Normalization strategies
- Classification thresholds
- Weight engine parameters
- Gemini API settings

## Data Requirements

The system expects a `data-sources` folder in the parent directory containing:
- Emission data (annual-co2-emission.csv)
- Coal consumption data
- Mining area data
- Policy timeline data
- Ownership information
- And other datasets as specified in the runbook

## Output Artifacts

Each run generates:
- **Scores**: JSON/CSV with pillar scores and classifications
- **Facts**: Structured data for LLM consumption
- **Logs**: Run metadata, weights audit, QA reports

## Phase 1 Scope

This implementation covers the MVP for 5 countries with 4 core pillars. Economic, Scarcity, and Infrastructure pillars are planned for Phase 2.

---

For detailed implementation specifications, see `docs/RUNBOOK_PHASE1.md`.
