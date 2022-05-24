"""
Project-specific metadata for global access.
Author: JiaWei Jiang
"""
# Primary key
PK = ["Date", "Capacity"]
# Auxiliary identifiers
TID = "Date"
SID = "ID"  # Sample identifier in testing set
# Originally provided column names with self-defined ordering
COLS = [
    "ID",
    "Date",
    "Lat",
    "Lon",
    "Irradiance",
    "Temp",
    "Module",
    "Capacity",
    "Angle",
    "Irradiance_m",
    "Temp_m",
    "Generation",
]
# Target column
TARGET = "Generation"
# `Temp` station
TEMP_STA = {
    "xx": [246.4, 492.8],  # Xain Xi 線西
    "xs": [314.88, 352, 99.84, 99.2, 267.52, 278.4, 343.2, 498.56],  # Xiu Shui 秀水
    "xw": [498.6],  # Xin Wu 新屋
    "tya": [283.2],  # Taoyuan agriculture 桃園農改場
    "lz": [438.3, 499.8],  # Lu Zhu 蘆竹
}
# `Irradiance` station
IRRA_STA = {
    "tca": [
        246.4,
        492.8,
        314.88,
        352,
        99.84,
        99.2,
        267.52,
        278.4,
        343.2,
        498.56,
    ],  # Taichung agriculture 台中農改場
    "tya": [498.6, 283.2],  # Taoyuan agriculture 桃園農改場
    "aic": [438.3, 499.8],  # Agriculture-Industrial Center 農工中心
}
# Generator module metadata
MODULE_META = {
    "Pmax": {
        "MM60-6RT-300": 300,
        "AUO PM060MW3 320W": 295,
        "SEC-6M-60A-295": 320,
        "AUO PM060MW3 325W": 325,
    },
    "Vmp": {
        "MM60-6RT-300": 32.61,
        "AUO PM060MW3 320W": 31.6,
        "SEC-6M-60A-295": 33.48,
        "AUO PM060MW3 325W": 33.66,
    },
    "Imp": {
        "MM60-6RT-300": 9.2,
        "AUO PM060MW3 320W": 9.34,
        "SEC-6M-60A-295": 9.56,
        "AUO PM060MW3 325W": 9.66,
    },
    "Voc": {
        "MM60-6RT-300": 38.97,
        "AUO PM060MW3 320W": 39.4,
        "SEC-6M-60A-295": 40.9,
        "AUO PM060MW3 325W": 41.1,
    },
    "Isc": {
        "MM60-6RT-300": 9.68,
        "AUO PM060MW3 320W": 9.85,
        "SEC-6M-60A-295": 10.24,
        "AUO PM060MW3 325W": 10.35,
    },
    "Eff": {
        "MM60-6RT-300": 18.44,
        "AUO PM060MW3 320W": 17.74,
        "SEC-6M-60A-295": 19.2,
        "AUO PM060MW3 325W": 19.5,
    },
}
