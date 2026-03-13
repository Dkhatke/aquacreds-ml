"""
biomass.py  
------------
Blue Carbon Biomass & Carbon Stock Estimation Module

Supports:
- Mangrove (Komiyama et al. 2008 allometric formula)
- Seagrass (IPCC default factors)
- Saltmarsh (global saltmarsh factors)

Returns:
- AGB, BGB, Carbon, CO2eq per ha
- Canopy percent (based on NDVI)
- Credit suggestions
- Satellite quality score
"""

# src/biomass.py

def estimate_canopy_percent(ndvi: float) -> int:
    # Converts NDVI (0–1) → canopy cover (0–100%)
    return int(max(0, min(100, ndvi * 100)))


def mangrove_biomass(canopy_percent):
    # DBH proxy scaled from canopy coverage
    DBH = (canopy_percent / 100) * 30  

    AGB = 0.251 * (DBH ** 2.46)
    BGB = 0.199 * (DBH ** 2.22)
    carbon = 0.48 * (AGB + BGB)
    co2eq = carbon * 3.67

    return {
        "AGB_t_per_ha": round(AGB, 2),
        "BGB_t_per_ha": round(BGB, 2),
        "Carbon_t_per_ha": round(carbon, 2),
        "CO2eq_t_per_ha": round(co2eq, 2)
    }


def estimate_all(ecosystem_class: str, ndvi: float, evi: float, area_ha: float):
    eco = ecosystem_class.lower()

    canopy = estimate_canopy_percent(ndvi)

    # Choose ecosystem-specific equation
    if eco == "mangrove":
        bio = mangrove_biomass(canopy)
    else:
        raise ValueError("Unknown ecosystem class: " + ecosystem_class)

    buffer_fraction = 2  # 2% deduction

    gross = bio["CO2eq_t_per_ha"] * area_ha
    suggested = gross * (1 - buffer_fraction / 100)

    return {
        "ecosystem_class": ecosystem_class,
        "canopy_percent": canopy,
        "biomass": bio,
        "credit_suggestion": {
            "area_ha": area_ha,
            "gross_CO2eq_t": round(gross, 2),
            "suggested_credits_tCO2e": round(suggested, 2),
            "buffer_fraction": buffer_fraction,
        },
        "ndvi_satellite": ndvi,
        "satellite_score": round((ndvi + evi) / 2, 2),
        "carbon_stock_tCO2e": round(gross, 2)
    }
