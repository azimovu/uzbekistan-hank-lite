# Source Data Provenance and Instructions

## 1. World Bank L2CU v03 Microdata (Download Separately)

Due to World Bank Microdata Library licensing terms, the raw household and individual survey archive is **not** redistributed in this repository.

To reproduce all outputs from scratch:

1. Register/log in at the World Bank Microdata Library:
   <https://microdata.worldbank.org/index.php/catalog/6412>
2. Download **Listening to Citizens of Uzbekistan Survey 2018–2025, v03 (CSV format)**:
   - Expected filename: `UZB_2018-2025_L2CU_v03_M_CSV.zip`
   - Reference SHA-256: `a7c96abf173e06ab358fdccda5988574e7f03b06096fe509d5f90448e57b39ac`
3. Place `UZB_2018-2025_L2CU_v03_M_CSV.zip` directly in this `data/` directory (or set `DATA_DIR=/path/to/data`).

## 2. Official Monthly Consumer Price Index (Included)

Open public data downloaded from the Statistics Agency under the President of the Republic of Uzbekistan (SIAT / SDMX Open Data Portal, <https://siat.stat.uz>):

| File | Coverage | Table | SHA-256 |
| :--- | :--- | :--- | :--- |
| `sdmx_cpi_1285.csv` | Monthly MoM CPI, 2010–2020 | Table 1285 | `33f2a5568b1de6181fd887c123cadf5efd029faab25397d84fa12b1960f48ddb` |
| `sdmx_cpi_1286.csv` | Monthly MoM CPI, 2021–2026 | Table 1286 | `38ce12ab3972e5da589085b4d0e69b6d28a667eb4e44ae79b1c8763e8085ade7` |
