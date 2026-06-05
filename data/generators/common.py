"""
common.py
Shared constants, lookup tables, and utility functions for all HEC data generators.
"""

import math
import random
from datetime import date, datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Date range
# ---------------------------------------------------------------------------
DATE_START = date(2022, 1, 1)
DATE_END = date(2026, 6, 1)

# ---------------------------------------------------------------------------
# HEC Facilities
# ---------------------------------------------------------------------------
# Each entry: name, country, lat, lon, capacity_m3_yr
FACILITIES = {
    "PIR": {
        "name": "Piraeus",
        "country": "GR",
        "lat": 37.9475,
        "lon": 23.6370,
        "capacity_m3_yr": 150_000,
    },
    "HAM": {
        "name": "Hamburg",
        "country": "DE",
        "lat": 53.5461,
        "lon": 9.9660,
        "capacity_m3_yr": 80_000,
    },
    "GIB": {
        "name": "Gibraltar",
        "country": "GI",
        "lat": 36.1408,
        "lon": -5.3536,
        "capacity_m3_yr": 60_000,
    },
    "MLT": {
        "name": "Malta",
        "country": "MT",
        "lat": 35.8989,
        "lon": 14.5146,
        "capacity_m3_yr": 40_000,
    },
}

# ---------------------------------------------------------------------------
# Ports (14 operational ports)
# ---------------------------------------------------------------------------
PORTS = {
    "PIR": {"name": "Piraeus",     "country": "GR", "lat": 37.9475, "lon":  23.6370},
    "HAM": {"name": "Hamburg",     "country": "DE", "lat": 53.5461, "lon":   9.9660},
    "GIB": {"name": "Gibraltar",   "country": "GI", "lat": 36.1408, "lon":  -5.3536},
    "MLT": {"name": "Malta",       "country": "MT", "lat": 35.8989, "lon":  14.5146},
    "GEN": {"name": "Genoa",       "country": "IT", "lat": 44.4056, "lon":   8.9463},
    "MRS": {"name": "Marseille",   "country": "FR", "lat": 43.3462, "lon":   5.3200},
    "BCN": {"name": "Barcelona",   "country": "ES", "lat": 41.3590, "lon":   2.1685},
    "RTM": {"name": "Rotterdam",   "country": "NL", "lat": 51.9054, "lon":   4.4661},
    "ALG": {"name": "Algeciras",   "country": "ES", "lat": 36.1272, "lon":  -5.4427},
    "LIM": {"name": "Limassol",    "country": "CY", "lat": 34.6741, "lon":  33.0379},
    "ALE": {"name": "Alexandria",  "country": "EG", "lat": 31.1842, "lon":  29.8763},
    "IST": {"name": "Istanbul",    "country": "TR", "lat": 41.0053, "lon":  28.9770},
    "PSD": {"name": "Port Said",   "country": "EG", "lat": 31.2565, "lon":  32.2841},
    "TAN": {"name": "Tangier",     "country": "MA", "lat": 35.7850, "lon":  -5.8029},
}

# ---------------------------------------------------------------------------
# Waste Categories
# ---------------------------------------------------------------------------
# Ranges expressed as (min_pct, max_pct); they are approximate, not strict sums.
# "typical" is the midpoint used for quick estimates.
WASTE_CATEGORIES = {
    "Bilge Water": {
        "oil_pct_range":    (3,  15),
        "water_pct_range":  (80, 95),
        "solids_pct_range": (1,   5),
        "code": "BW",
    },
    "Fuel Oil Sludge": {
        "oil_pct_range":    (20, 60),
        "water_pct_range":  (10, 40),
        "solids_pct_range": (10, 30),
        "code": "FOS",
    },
    "Oily Tank Washings": {
        "oil_pct_range":    (30, 70),
        "water_pct_range":  (20, 60),
        "solids_pct_range": (2,  10),
        "code": "OTW",
    },
    "Dirty Ballast Water": {
        "oil_pct_range":    (0.1,  3),
        "water_pct_range":  (95,  99),
        "solids_pct_range": (0.1,  1),
        "code": "DBW",
    },
    "Scale & Sludge from Tank Cleaning": {
        "oil_pct_range":    (10, 30),
        "water_pct_range":  (10, 30),
        "solids_pct_range": (40, 70),
        "code": "SSTC",
    },
    "Cargo Residues (Crude)": {
        "oil_pct_range":    (70, 95),
        "water_pct_range":  (3,  20),
        "solids_pct_range": (2,  10),
        "code": "CRC",
    },
    "Cargo Residues (Product)": {
        "oil_pct_range":    (80, 98),
        "water_pct_range":  (1,  10),
        "solids_pct_range": (0.5, 3),
        "code": "CRP",
    },
    "Offshore Production Slops": {
        "oil_pct_range":    (40, 70),
        "water_pct_range":  (20, 50),
        "solids_pct_range": (5,  20),
        "code": "OPS",
    },
    "Engine Room Oily Water": {
        "oil_pct_range":    (2,  10),
        "water_pct_range":  (85, 97),
        "solids_pct_range": (1,   5),
        "code": "EROW",
    },
    "Vegetable Oil Residues": {
        "oil_pct_range":    (60, 90),
        "water_pct_range":  (5,  30),
        "solids_pct_range": (2,  10),
        "code": "VOR",
    },
    "Chemical Washing Residues": {
        "oil_pct_range":    (10, 50),
        "water_pct_range":  (30, 70),
        "solids_pct_range": (5,  20),
        "code": "CWR",
    },
}

# ---------------------------------------------------------------------------
# Vessel Types
# ---------------------------------------------------------------------------
VESSEL_TYPES = {
    "Crude Oil Tanker": {
        "gt_range": (80_000, 320_000),
        "engine_kw_range": (18_000, 35_000),
        "waste_factor": 1.8,   # multiplier on baseline waste volumes
    },
    "Product Tanker": {
        "gt_range": (5_000, 60_000),
        "engine_kw_range": (4_000, 14_000),
        "waste_factor": 1.4,
    },
    "Chemical Tanker": {
        "gt_range": (3_000, 40_000),
        "engine_kw_range": (3_000, 10_000),
        "waste_factor": 1.6,
    },
    "LNG Carrier": {
        "gt_range": (70_000, 180_000),
        "engine_kw_range": (20_000, 40_000),
        "waste_factor": 0.9,
    },
    "Bulk Carrier": {
        "gt_range": (10_000, 200_000),
        "engine_kw_range": (6_000, 20_000),
        "waste_factor": 1.0,
    },
    "Container Ship": {
        "gt_range": (10_000, 220_000),
        "engine_kw_range": (20_000, 80_000),
        "waste_factor": 1.1,
    },
    "General Cargo": {
        "gt_range": (1_000, 20_000),
        "engine_kw_range": (800, 8_000),
        "waste_factor": 0.9,
    },
    "Ro-Ro / Ferry": {
        "gt_range": (3_000, 60_000),
        "engine_kw_range": (5_000, 40_000),
        "waste_factor": 1.2,
    },
    "Offshore Supply Vessel": {
        "gt_range": (500, 5_000),
        "engine_kw_range": (1_500, 8_000),
        "waste_factor": 1.5,
    },
    "Fishing Vessel": {
        "gt_range": (50, 2_000),
        "engine_kw_range": (150, 3_000),
        "waste_factor": 0.7,
    },
}

# ---------------------------------------------------------------------------
# Categorical lists with probability weights
# ---------------------------------------------------------------------------

FUEL_TYPES = ["HFO", "VLSFO", "MGO", "MDO", "LNG", "Methanol"]
FUEL_TYPE_WEIGHTS = [0.30, 0.35, 0.15, 0.12, 0.06, 0.02]

COLLECTION_METHODS = [
    "Direct Pump Transfer",
    "Vacuum Tanker",
    "Portable Tank",
    "Ship's Own Pump",
    "Gravity Feed",
]
COLLECTION_METHOD_WEIGHTS = [0.45, 0.25, 0.15, 0.10, 0.05]

MIXING_LEVELS = ["None", "Low", "Medium", "High"]
MIXING_LEVEL_WEIGHTS = [0.20, 0.35, 0.30, 0.15]

CONTRACT_TYPES = ["Spot", "Annual", "Quarterly", "Framework", "Emergency"]
CONTRACT_TYPE_WEIGHTS = [0.30, 0.40, 0.15, 0.10, 0.05]

PRE_TREATMENTS = [
    "None",
    "Heating",
    "Gravity Separation",
    "Centrifugation",
    "Chemical Demulsification",
    "Filtration",
]
PRE_TREATMENT_WEIGHTS = [0.30, 0.20, 0.18, 0.15, 0.10, 0.07]

TARGET_SPECS = [
    "MARPOL Annex I",
    "IMO 2020 Compliant",
    "EU Waste Directive",
    "Recovered Oil Grade A",
    "Recovered Oil Grade B",
    "Fuel Blend Feedstock",
    "Re-refinery Feedstock",
]
TARGET_SPEC_WEIGHTS = [0.28, 0.20, 0.18, 0.12, 0.10, 0.07, 0.05]

BATCH_PRIORITIES = ["Normal", "High", "Urgent", "Scheduled", "Deferred"]
BATCH_PRIORITY_WEIGHTS = [0.50, 0.20, 0.10, 0.15, 0.05]

SEPARATOR_TYPES = [
    "3-Phase Disc Stack Centrifuge",
    "Decanter Centrifuge",
    "Gravity Separator",
    "Plate Separator",
    "DAF Unit",
    "Heating Unit",
]
SEPARATOR_TYPE_WEIGHTS = [0.25, 0.20, 0.20, 0.15, 0.12, 0.08]

# ---------------------------------------------------------------------------
# Vessel names
# ---------------------------------------------------------------------------

HEC_VESSEL_NAMES = [
    "HEC Aegean Star",
    "HEC Apollo",
    "HEC Artemis",
    "HEC Athena",
    "HEC Atlas",
    "HEC Boreas",
    "HEC Calypso",
    "HEC Charon",
    "HEC Delphi",
    "HEC Europa",
    "HEC Helios",
    "HEC Hermes",
    "HEC Hydra",
    "HEC Iolcos",
    "HEC Iris",
    "HEC Kronos",
    "HEC Lethe",
    "HEC Medusa",
    "HEC Nereid",
    "HEC Nereus",
    "HEC Oceanus",
    "HEC Poseidon",
    "HEC Selene",
    "HEC Tethys",
    "HEC Triton",
]

CLIENT_VESSEL_NAMES = [
    "MSC Adriatica",
    "MSC Levante",
    "CMA Mistral",
    "Cosco Pacific Voyager",
    "Euronav Olympia",
    "Nordic Bulldog",
    "Frontline Panther",
    "Tsakos Apollon",
    "Minerva Helen",
    "Danaos Theseus",
    "Thenamaris Eagle",
    "Capital Perseus",
    "Maran Castor",
    "Neda Nymph",
    "Diana Voyager",
    "Scorpio Virgo",
    "Product Thetis",
    "Stena Baltica",
    "Carnival Harmony",
    "Louis Aura",
    "Eastern Pharos",
    "Arabian Star",
    "Pacific Cormorant",
    "Atlantic Kestrel",
    "Nordic Grace",
    "Adriatic Mariner",
    "Black Sea Pioneer",
    "Bosphorus Spirit",
]

# ---------------------------------------------------------------------------
# Equipment manufacturers and model numbers
# ---------------------------------------------------------------------------

EQUIPMENT_MANUFACTURERS = {
    "Alfa Laval": {
        "3-Phase Disc Stack Centrifuge": ["AFPX 510", "AFPX 614", "S816", "BRPX 617"],
        "Gravity Separator":             ["AlfaVap 800", "AlfaVap 1000"],
        "Plate Separator":               ["MR355", "MR455", "MR510"],
        "Heating Unit":                  ["AQUA 500", "AQUA 800"],
        "DAF Unit":                      ["ALDAF 500"],
        "Decanter Centrifuge":           ["ALDEC G2-415", "ALDEC G2-515"],
    },
    "GEA Westfalia": {
        "3-Phase Disc Stack Centrifuge": ["OSE 40-01-067", "OSE 80-01-067", "UCD 305"],
        "Decanter Centrifuge":           ["CA 365-1", "CA 520-2"],
        "Gravity Separator":             ["GS 310", "GS 410"],
        "DAF Unit":                      ["FloDaf 300"],
        "Plate Separator":               ["PL 350", "PL 450"],
        "Heating Unit":                  ["HEX 200", "HEX 400"],
    },
    "Flottweg": {
        "Decanter Centrifuge":           ["Z4E-4/441", "Z5E-4/441", "Z6E-4/441"],
        "3-Phase Disc Stack Centrifuge": ["Tricanter Z4D-4", "Tricanter Z5D-4"],
        "Plate Separator":               ["SCD 360", "SCD 460"],
        "Gravity Separator":             ["FGS 200", "FGS 400"],
        "Heating Unit":                  ["FHU 300"],
        "DAF Unit":                      ["FDAF 200"],
    },
    "Andritz": {
        "Decanter Centrifuge":           ["D4L-H", "D5L-H", "D6L-H"],
        "3-Phase Disc Stack Centrifuge": ["SX 35T", "SX 50T"],
        "Gravity Separator":             ["AGS 300", "AGS 500"],
        "DAF Unit":                      ["HydroFloat 400"],
        "Plate Separator":               ["APL 300", "APL 450"],
        "Heating Unit":                  ["AHU 250", "AHU 500"],
    },
    "Pieralisi": {
        "3-Phase Disc Stack Centrifuge": ["LEOPARD 3", "LEOPARD 5", "GIANT 4"],
        "Decanter Centrifuge":           ["JUMBO 2 50", "JUMBO 3 50"],
        "Gravity Separator":             ["PGS 250"],
        "Plate Separator":               ["PPL 350"],
        "DAF Unit":                      ["PDAF 300"],
        "Heating Unit":                  ["PHU 400"],
    },
}

# ---------------------------------------------------------------------------
# Helper: seasonal demand factor
# ---------------------------------------------------------------------------

def seasonal_factor(date_val: date, port_code: str) -> float:
    """
    Return a multiplicative seasonal demand factor (0.7 – 1.4) for a given
    date and port, reflecting shipping traffic patterns.

    - Mediterranean ports peak in summer (tourist season, bulk shipping).
    - Northern European ports (HAM, RTM) peak in spring / autumn; quieter in
      deep winter.
    - Gibraltar (GIB) is relatively flat (transit port).
    """
    month = date_val.month if isinstance(date_val, (date, datetime)) else date_val

    northern_eu = {"HAM", "RTM"}
    transit = {"GIB", "TAN", "ALG"}

    if port_code in northern_eu:
        # Sinusoidal peaking around month 9 (September)
        base = 0.85 + 0.30 * math.sin(math.pi * (month - 3) / 6)
    elif port_code in transit:
        # Nearly flat with slight summer dip (weather-independent)
        base = 1.0 + 0.08 * math.sin(math.pi * (month - 6) / 6)
    else:
        # Mediterranean: peak summer (July = 1.35)
        base = 0.85 + 0.50 * math.sin(math.pi * (month - 3) / 7)

    return round(max(0.70, min(1.40, base)), 4)


# ---------------------------------------------------------------------------
# Helper: ambient / sea-water temperature for a facility
# ---------------------------------------------------------------------------

_FACILITY_TEMP_PARAMS = {
    # (mean_annual_C, amplitude_C)  — sinusoidal, coldest in Jan, warmest in Jul
    "PIR": (17.5, 9.0),
    "HAM": (10.0, 11.0),
    "GIB": (17.0, 7.5),
    "MLT": (18.5, 8.0),
}


def temperature_for_location(date_val: date, facility_code: str) -> float:
    """
    Return estimated ambient/seawater temperature (°C) for the given facility
    and date using a simple sinusoidal model.
    """
    month = date_val.month if isinstance(date_val, (date, datetime)) else date_val
    mean, amp = _FACILITY_TEMP_PARAMS.get(facility_code, (15.0, 8.0))
    # Coldest ~January (month 1), warmest ~July (month 7)
    temp = mean + amp * math.sin(math.pi * (month - 1) / 6 - math.pi / 2)
    return round(temp, 2)


# ---------------------------------------------------------------------------
# Helper: Walther viscosity-temperature equation
# ---------------------------------------------------------------------------

def walther_viscosity(viscosity_40c: float, temperature_c: float) -> float:
    """
    Estimate kinematic viscosity (cSt) at *temperature_c* given *viscosity_40c*
    using the ASTM D341 / Walther equation.

    W(T) = log10(log10(v + 0.7)) = A - B * log10(T_K)
    where T_K = temperature + 273.15.

    Parameters
    ----------
    viscosity_40c : float
        Known kinematic viscosity at 40 °C (cSt).
    temperature_c : float
        Target temperature (°C).

    Returns
    -------
    float
        Estimated kinematic viscosity at *temperature_c* (cSt).
    """
    T1_K = 40.0 + 273.15
    T2_K = temperature_c + 273.15

    v1 = viscosity_40c + 0.7
    W1 = math.log10(math.log10(v1))

    # Reference second point at 100 °C (assuming VI ~ 95 mineral oil):
    # Use a fixed slope consistent with typical marine fuel oil
    # B = (W1 - W2) / (log10(T2_K) - log10(T1_K)) → empirical B ≈ 3.7 for HFO
    B = 3.7

    A = W1 + B * math.log10(T1_K)
    W2 = A - B * math.log10(T2_K)

    # Invert:  v = 10^(10^W2) - 0.7
    viscosity = 10 ** (10 ** W2) - 0.7
    return round(max(0.5, viscosity), 4)


# ---------------------------------------------------------------------------
# Helper: generate a plausible IMO number
# ---------------------------------------------------------------------------

def generate_imo_number(rng: Optional[random.Random] = None) -> str:
    """
    Generate a syntactically valid IMO ship identification number.

    The check digit is the last digit, computed as:
        sum(digit_i * weight_i for i, weight in zip(digits[:6], [7,6,5,4,3,2])) mod 10
    The prefix 9 digits start with 9 (the common ship prefix).
    """
    if rng is None:
        rng = random.Random()

    while True:
        # IMO numbers for ships typically start with 7, 8, or 9
        prefix = rng.choice([7, 8, 9])
        d = [prefix] + [rng.randint(0, 9) for _ in range(5)]
        check = sum(v * w for v, w in zip(d, [7, 6, 5, 4, 3, 2])) % 10
        imo = "IMO" + "".join(str(x) for x in d) + str(check)
        return imo


# ---------------------------------------------------------------------------
# Helper: Haversine sea distance (great-circle approximation, in nautical miles)
# ---------------------------------------------------------------------------

_R_NM = 3440.065  # Earth radius in nautical miles


def haversine_distance_nm(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    Return the great-circle distance in nautical miles between two
    (lat, lon) coordinates expressed in decimal degrees.

    Note: this gives the straight-line (over water) approximation.
    Actual sea routes may be longer due to landmasses.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(_R_NM * c, 1)
