# FVI Phase 1 Canonical Schema

This document defines the exact table schemas and fields for the FVI system Phase 1.
All tables are country-year unless marked "country-only." 
Key constraint: iso3 ∈ {IND, CHN, USA, JPN, ZAF}

## Tables

### country_dim (country-only)
Primary key: iso3

| Field | Type | Constraints |
|-------|------|-------------|
| iso3 | TEXT | PK, ∈ {IND, CHN, USA, JPN, ZAF} |
| name | TEXT | Country name |

### fact_emissions

| Field | Type | Constraints |
|-------|------|-------------|
| iso3 | TEXT | FK to country_dim, ∈ {IND, CHN, USA, JPN, ZAF} |
| year | INTEGER | Reference year |
| emissions_tco2e | FLOAT | Total CO2 emissions in tonnes |
| intensity_tco2_per_mwh | FLOAT | Nullable, CO2 tonnes per MWh |
| coverage_pct | FLOAT | Nullable, 0-100 |
| data_quality | FLOAT | 0-1 scale |
| dataset_ids | ARRAY | Source dataset identifiers |

### fact_necessity

| Field | Type | Constraints |
|-------|------|-------------|
| iso3 | TEXT | FK to country_dim, ∈ {IND, CHN, USA, JPN, ZAF} |
| year | INTEGER | Reference year |
| coal_share_electricity_pct | FLOAT | 0-100 |
| coal_consumption_twh | FLOAT | TWh consumption |
| population | FLOAT | Nullable, total population |
| pop_reliant_pct | FLOAT | Nullable, 0-100 |
| data_quality | FLOAT | 0-1 scale |
| dataset_ids | ARRAY | Source dataset identifiers |

### fact_ecological (country-only)

| Field | Type | Constraints |
|-------|------|-------------|
| iso3 | TEXT | FK to country_dim, ∈ {IND, CHN, USA, JPN, ZAF} |
| mining_area_km2 | FLOAT | Mining area in km² |
| mining_sites_count | INTEGER | Number of mining sites |
| deforest_area_km2 | FLOAT | Nullable, deforestation area |
| ash_sites_count | INTEGER | Nullable, coal ash sites |
| ash_incidents_count | INTEGER | Nullable, ash incidents |
| activity_denominator_unit | TEXT | Unit for normalization |
| activity_denominator_value | FLOAT | Denominator value |
| data_quality | FLOAT | 0-1 scale |
| dataset_ids | ARRAY | Source dataset identifiers |

### fact_support

| Field | Type | Constraints |
|-------|------|-------------|
| iso3 | TEXT | FK to country_dim, ∈ {IND, CHN, USA, JPN, ZAF} |
| year | INTEGER | Reference year |
| exit_year | INTEGER | Nullable, planned coal exit year |
| coverage_pct | FLOAT | Nullable, 0-100 |
| state_ownership_share_pct | FLOAT | Nullable, 0-100 |
| subsidy_proxy_idx | FLOAT | Nullable, 0-1 |
| data_quality | FLOAT | 0-1 scale |
| dataset_ids | ARRAY | Source dataset identifiers |

## Notes

- Economic, Scarcity, and Infrastructure pillars are placeholders in Phase 1
- All nullable fields should be handled gracefully when missing
- Data quality scores reflect completeness and methodology confidence
- Dataset IDs enable full provenance tracking
