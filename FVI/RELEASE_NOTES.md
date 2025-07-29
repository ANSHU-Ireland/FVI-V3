# FVI Phase 1 Release Notes

## Version 1.0.0 - Initial MVP Release

**Release Date:** July 29, 2025

### 🎯 **Phase 1 Scope**
Complete MVP implementation for **5 target countries**: India (IND), China (CHN), United States (USA), Japan (JPN), and South Africa (ZAF).

### ✨ **Key Features**

#### **Core System Architecture**
- **6 Core Modules**: Complete modular architecture
- **Configuration-Driven**: YAML-based configuration system
- **Data Pipeline**: Automated ingestion from data-sources folder
- **Quality Assurance**: Comprehensive validation and QA gates

#### **Pillar Assessment System**
- **Emissions**: Absolute emissions, intensity, coverage gaps
- **Necessity**: Coal dependency, consumption lock-in, population at risk  
- **Ecological**: Land disturbance, site density, deforestation exposure
- **Artificial Support**: Exit penalties, state ownership, coverage gaps

#### **Dynamic Weighting Engine**
- **Persona-Driven**: Investor/Regulator/Operator specific adjustments
- **Depth-Aware**: Quick/Standard/Deep analysis scaling
- **Time Horizon**: 0-5y/5-10y/10-30y considerations
- **Risk & Capital**: Risk tolerance and capex capacity factors

#### **Classification System**
- **3-Tier Categories**: Sustainable / Critical Transition / Decommission
- **Guardrail Rules**: Policy constraint enforcement
- **Borderline Detection**: Threshold proximity flagging

#### **AI-Powered Explanations**
- **Gemini Integration**: Google's Gemini-2.0-flash model
- **Grounded Facts**: Structured fact-based explanations
- **Personalized**: User profile-driven recommendations
- **Data Gap Handling**: Clarifying questions for missing data

### 📊 **Technical Implementation**

#### **Data Processing**
- **Country Filtering**: Strict enforcement of 5-country scope
- **Reference Year**: Deterministic selection algorithm
- **Quality Scoring**: Data quality tracking and penalties
- **Winsorization**: Outlier handling at 1st/99th percentiles

#### **Validation Gates**
- **Schema Validation**: Required column presence checks
- **Type/Range Validation**: Data type and value range enforcement
- **Cross-Checks**: Consistency validation between metrics
- **Provenance Tracking**: Dataset source tracking for all metrics

#### **Output Artifacts**
- **Scores**: JSON/CSV with pillar scores and classifications
- **Facts**: Structured data for AI consumption
- **QA Reports**: Comprehensive validation and lineage logs
- **Run Metadata**: Complete audit trail for reproducibility

### 🚀 **Getting Started**

```bash
# Clone the repository
git clone https://github.com/ANSHU-Ireland/FVI-V3.git
cd FVI-V3

# Setup and install dependencies
python setup.py

# Run an assessment
python main.py
```

### 📋 **Requirements**
- Python 3.8+
- Data sources folder in parent directory (`../data-sources`)
- Gemini API key (configured in `config/config.yaml`)

### 🔧 **Configuration**
All system parameters are configurable via `config/config.yaml`:
- Reference year selection policy
- Normalization strategies per metric
- Classification thresholds
- Weight engine parameters
- Gemini API settings

### 📈 **Phase 2 Roadmap**
- Economic pillar implementation
- Scarcity pillar implementation  
- Infrastructure pillar implementation
- Enhanced within-pillar subweight nudging
- Sensitivity analysis and scenario modeling

### 🐛 **Known Limitations**
- Phase 1 scope limited to 4 pillars for 5 countries
- Economic/Scarcity/Infrastructure pillars marked as N/A
- Manual micro-table population required for some metrics

### 📞 **Support**
For issues, questions, or contributions, please use the GitHub Issues page.
