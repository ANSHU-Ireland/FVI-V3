# FVI Phase 1 - Fossil Vulnerability Index

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub release](https://img.shields.io/github/release/ANSHU-Ireland/FVI-V3.svg)](https://github.com/ANSHU-Ireland/FVI-V3/releases)

A comprehensive assessment system for evaluating fossil fuel vulnerability across five target countries: India (IND), China (CHN), United States (USA), Japan (JPN), and South Africa (ZAF).

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ANSHU-Ireland/FVI-V3.git
   cd FVI-V3
   ```

2. **Setup the environment:**
   ```bash
   python setup.py
   ```

3. **Run the assessment:**
   ```bash
   python main.py
   ```

4. **Follow the interactive prompts** to provide your profile and get a personalized assessment.

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

## 📊 Sample Output

```
FVI Phase 1 - Fossil Vulnerability Index Assessment
=======================================================
Country: IND
Reference Year: 2022
Composite Score: 72.3
Classification: Critical Transition

Pillar Scores:
  Emissions: 78.5 (weight: 0.28)
  Necessity: 85.2 (weight: 0.32)  
  Ecological: 65.1 (weight: 0.22)
  ArtificialSupport: 68.9 (weight: 0.18)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ANSHU-Ireland/FVI-V3/issues)
- **Documentation**: See `docs/` folder for detailed specifications
- **Release Notes**: [RELEASE_NOTES.md](RELEASE_NOTES.md)

---

For detailed implementation specifications, see `docs/RUNBOOK_PHASE1.md`.
