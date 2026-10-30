"""Download NASA POWER hourly irradiance and temperature (2001-2024) for the experiments.

Usage: python experiments/fetch_weather.py <out_dir> [latitude] [longitude]
Default point: Valencia area, Venezuela (10.16 N, 68.00 W). Solar data come from CERES SYN1deg on a 1 degree
grid, so this is a regional series, not a site measurement. Times are local solar time (LST).
"""

import sys
import time
import urllib.request
from pathlib import Path

URL = (
    "https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN,T2M&community=RE"
    "&longitude={lon}&latitude={lat}&start={year}0101&end={year}1231&format=JSON&time-standard=LST"
)
YEARS = range(2001, 2025)  # POWER hourly starts on 1 January 2001
PAUSE_S = 2  # be polite to the public API


def main() -> None:
    out = Path(sys.argv[1])
    lat = sys.argv[2] if len(sys.argv) > 2 else "10.16"
    lon = sys.argv[3] if len(sys.argv) > 3 else "-68.00"
    out.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        target = out / f"p{year}.json"
        if target.exists():
            continue
        with urllib.request.urlopen(URL.format(lat=lat, lon=lon, year=year), timeout=120) as resp:
            target.write_bytes(resp.read())
        print(f"{year} -> {target}")
        time.sleep(PAUSE_S)


if __name__ == "__main__":
    main()
