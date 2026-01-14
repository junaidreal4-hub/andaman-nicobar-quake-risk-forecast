from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    raw_csv_path: str = "data/raw/andaman_usgs_events.csv"
