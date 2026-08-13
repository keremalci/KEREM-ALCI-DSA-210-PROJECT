"""
Shared data-cleaning utilities for the DSA210 Kurtkoy Rental Project.

This module is the SINGLE source of truth for parsing the three raw data
sources (Sahibinden, Emlakjet, Hepsiemlak). Both `test_results` and
`analysis/ml_analysis.py` import from here instead of each keeping their own
(previously inconsistent, occasionally buggy) copies of these functions.

Why this file exists (bugs it fixes vs. the original project):
- The old `ml_analysis.ipynb` used a single generic `clean_numeric()`
  for BOTH `Rooms` and `Building Age`. For Rooms, `"4+2".replace('+1', '')`
  does nothing (no literal "+1" substring), so `float("4+2".split()[0])`
  raises and silently becomes NaN - losing real rows. For Rooms that DO
  contain "+1" (e.g. "3+1"), the same code strips it to "3", silently
  dropping the living-room count instead of summing it like the older
  `test_results.ipynb` cleaner did.
- For Building Age, values are Turkish text buckets ("6-10 arasi", "Sifir
  Bina", "Belirtilmemis") for Sahibinden, and are entirely missing (NaN) for
  Emlakjet and Hepsiemlak in the real data. `clean_numeric()` can't parse
  any of that, so the feature was effectively unusable/silently broken.
  Building Age is now treated as a categorical bucket (with an explicit
  "Unknown" bucket) instead of a fabricated numeric column.
- Hepsiemlak's `Rooms` column is a stringified Python list, e.g. "['2+1']"
  and its `Area(m2)` column is a stringified dict, e.g.
  "{'netSqm': 80, 'grossSqm': [95], ...}". Neither of the old cleaners in
  the two notebooks handled both of these AND Sahibinden's plain int AND
  Emlakjet's "138 m2" string in one consistent way.
"""
from __future__ import annotations

import ast
import re
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference source files (relative to project root)
# ---------------------------------------------------------------------------
SOURCE_FILES = {
    "Emlakjet": "data/raw/emlakjet/emlakjet_listings.xlsx",
    "Sahibinden": "data/raw/sahibinden/sahibinden_enriched_listings.xlsx",
    "Hepsiemlak": "data/raw/hepsiemlak/hepsiemlak_listings.xlsx",
}


def load_all_sources(base_dir: str = ".") -> pd.DataFrame:
    """Load all three raw files (if present), tag each row with its Source,
    and concatenate them into a single DataFrame. Missing files are skipped
    with a warning rather than raising, so the pipeline degrades gracefully.
    """
    frames = []
    for source, rel_path in SOURCE_FILES.items():
        path = os.path.join(base_dir, rel_path)
        if not os.path.exists(path):
            print(f"  [!] {source}: file not found at '{path}', skipping.")
            continue
        try:
            df = pd.read_excel(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [!] {source}: failed to read '{path}' ({exc}), skipping.")
            continue
        df["Source"] = source
        frames.append(df)
        print(f"  Loaded {len(df)} rows from {source} ({path})")

    if not frames:
        raise FileNotFoundError("No raw data files could be loaded from data/raw/*")

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Field-level cleaners (each one is source-agnostic and handles every
# format observed across Sahibinden / Emlakjet / Hepsiemlak)
# ---------------------------------------------------------------------------

def clean_price(val):
    """'35.000 TL' / '45.000 TL' / 32000 (already numeric) -> float TL/month."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).upper().replace("TL", "").replace(".", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_area(val):
    """Handles:
      - Sahibinden: already numeric / numeric-like string ('75')
      - Emlakjet:   '138 m2' (m-superscript-2)
      - Hepsiemlak: "{'netSqm': 80, 'grossSqm': [95], ...}" (stringified dict)
    Prefers grossSqm over netSqm for Hepsiemlak (consistent with Sahibinden's
    "m2 (Brut)" preference in the original scraper).
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()

    # Hepsiemlak: stringified dict
    if s.startswith("{") and ("Sqm" in s or "sqm" in s):
        try:
            d = ast.literal_eval(s)
            gross = d.get("grossSqm")
            if isinstance(gross, list) and gross:
                return float(gross[0])
            if isinstance(gross, (int, float)) and gross:
                return float(gross)
            net = d.get("netSqm")
            if net:
                return float(net)
        except (ValueError, SyntaxError):
            pass
        # Fallback to regex if literal_eval fails on malformed strings
        m = re.search(r"'grossSqm':\s*\[(\d+(?:\.\d+)?)\]", s)
        if m:
            return float(m.group(1))
        m = re.search(r"'netSqm':\s*(\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1))
        return np.nan

    # Standard "138 m2" / "138 m²" format
    s = s.replace("m²", "").replace("m2", "").replace(".", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_rooms(val):
    """Returns TOTAL room count (bedrooms + living room), matching how
    a renter would read "2+1" (2 bedrooms + 1 living room = 3 total rooms).

    Handles:
      - Sahibinden / Emlakjet: '2+1', '3+1'
      - Emlakjet studio:       '1 Oda'
      - Hepsiemlak:            "['2+1']" (stringified single-item list)
    """
    if pd.isna(val):
        return np.nan

    s = str(val)

    # Hepsiemlak: "['2+1']" -> unwrap the list first
    if s.strip().startswith("["):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list) and parsed:
                s = str(parsed[0])
        except (ValueError, SyntaxError):
            s = s.strip("[]'\" ")

    s = s.strip().lower()

    if "oda" in s:  # e.g. "1 oda" -> studio
        return 1.0

    if "+" in s:
        parts = s.split("+")
        try:
            return float(parts[0]) + float(parts[1])
        except (ValueError, IndexError):
            return np.nan

    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_bathrooms(val):
    """Sahibinden uses the literal string 'Yok' (= "none") for 0 bathrooms;
    Emlakjet has real numbers; Hepsiemlak doesn't provide this field at all
    (always NaN in the raw data) - that's a genuine data gap, not a bug.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    if s in ("yok", "none", ""):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_furnish(val):
    """Normalizes to 'Furnished' / 'Unfurnished' / 'Unknown'.
    Handles Python bools (Hepsiemlak), Turkish text (Sahibinden: Evet/Hayir),
    and Emlakjet's Bos/Esyali wording.
    """
    if pd.isna(val):
        return "Unknown"
    if isinstance(val, (bool, np.bool_)):
        return "Furnished" if val else "Unfurnished"

    s = str(val).strip().lower()
    if s in ("true", "evet", "esyali", "eşyalı"):
        return "Furnished"
    if s in ("false", "hayir", "hayır", "bos", "boş"):
        return "Unfurnished"
    return "Unknown"


def clean_listing_type(val):
    """Normalizes to 'Agent' / 'Owner' / 'Unknown'.
    NOTE: this field is only populated for Sahibinden in the real data
    (Emlakjet and Hepsiemlak leave it blank) - the Agent-vs-Owner hypothesis
    test is therefore inherently a Sahibinden-only comparison. We keep that
    limitation explicit rather than pretending it's a 3-source comparison.
    """
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip().lower()
    if "ofis" in s or "agent" in s or "emlak" in s:
        return "Agent"
    if "sahibinden" in s or "owner" in s or "sahib" in s:
        return "Owner"
    return "Unknown"


AGE_BUCKET_ORDER = ["0-5 Years", "6-10 Years", "11-15 Years", "16-20 Years", "21+ Years", "Unknown"]


def clean_age_bucket(val):
    """Buckets Building Age into consistent categories. Emlakjet and
    Hepsiemlak never populate this field in the real data -> 'Unknown' is a
    legitimate, expected bucket (not a parsing failure)."""
    if pd.isna(val):
        return "Unknown"
    s = str(val).strip()

    if s in ("Belirtilmemis", "Belirtilmemiş", ""):
        return "Unknown"
    if s in ("1", "2", "3", "4", "5", "0", "Sifir Bina", "Sıfır Bina"):
        return "0-5 Years"
    if "6-10" in s:
        return "6-10 Years"
    if "11-15" in s:
        return "11-15 Years"
    if "16-20" in s:
        return "16-20 Years"
    if "21-25" in s or "21 ve" in s.lower() or "21+" in s:
        return "21+ Years"

    # Numeric fallback (e.g. a raw integer year count outside 0-5)
    try:
        years = float(s)
        if years <= 5:
            return "0-5 Years"
        if years <= 10:
            return "6-10 Years"
        if years <= 15:
            return "11-15 Years"
        if years <= 20:
            return "16-20 Years"
        return "21+ Years"
    except ValueError:
        return "Unknown"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all field cleaners and returns a new DataFrame with
    consistently-named `clean_*` columns, ready for stats/ML use."""
    out = df.copy()
    out["clean_price"] = out["Price"].apply(clean_price)
    out["clean_area"] = out["Area(m2)"].apply(clean_area)
    out["clean_rooms"] = out["Rooms"].apply(clean_rooms)
    out["clean_bathrooms"] = out["Bathrooms"].apply(clean_bathrooms) if "Bathrooms" in out.columns else np.nan
    out["clean_furnish"] = out["Furnishment"].apply(clean_furnish) if "Furnishment" in out.columns else "Unknown"
    out["clean_listing_type"] = out["Listing Type"].apply(clean_listing_type) if "Listing Type" in out.columns else "Unknown"
    out["clean_age_bucket"] = out["Building Age"].apply(clean_age_bucket) if "Building Age" in out.columns else "Unknown"

    for dist_col in ["Distance to Metro (km)", "Distance to University (km)", "Distance to Bus Station (km)"]:
        if dist_col in out.columns:
            out[dist_col] = pd.to_numeric(out[dist_col], errors="coerce")

    out = out.dropna(subset=["clean_price"])
    return out


def flag_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Flags likely duplicate listings posted across multiple platforms.

    Heuristic: same rounded price (nearest 500 TL), same rounded area
    (nearest 5 m2), and same total room count. This isn't perfect (two
    genuinely different units can coincidentally match), so we only FLAG
    for review via a `Likely_Duplicate_Group` column rather than silently
    dropping rows - dropping automatically on a 47-row dataset is risky.
    """
    out = df.copy()
    out["_price_bucket"] = (out["clean_price"] / 500).round() * 500
    out["_area_bucket"] = (out["clean_area"] / 5).round() * 5

    group_cols = ["_price_bucket", "_area_bucket", "clean_rooms"]
    valid = out.dropna(subset=group_cols)

    dup_group_id = {}
    group_counter = 0
    for key, group in valid.groupby(group_cols):
        if len(group) > 1 and group["Source"].nunique() > 1:
            group_counter += 1
            for idx in group.index:
                dup_group_id[idx] = group_counter

    out["Likely_Duplicate_Group"] = out.index.map(dup_group_id.get)
    out = out.drop(columns=["_price_bucket", "_area_bucket"])
    return out
