import glob
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

# ── Filename patterns ──────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})")

_ZIP_PATTERNS = {
    'obs': re.compile(r"Reports/ObserverLog_Detailed_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.csv"),
    'pss': re.compile(r"Reports/PSS_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.csv"),
    'cog': re.compile(r"Reports/FinalCOG_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.csv"),
}

_FOLDER_PATTERNS = {
    'obs': re.compile(r"ObserverLog_Detailed_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.csv"),
    'pss': re.compile(r"PSS_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.csv"),
    'cog': re.compile(r"FinalCOG_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.csv"),
}


# ── Gather ─────────────────────────────────────────────────────────────────────

def gather_files(source_path, raw_dir, mode="zip", target_zip=None):
    """
    Extract or copy the three log CSVs (OBS, PSS, COG) into raw_dir/Day_YYYYMMDD/.

    Args:
        source_path: Directory containing the zip file or folder to scan.
        raw_dir:     Destination for raw copies, organised into Day_YYYYMMDD subfolders.
        mode:        'zip', 'folder', or 'both'.
        target_zip:  Zip filename (required when mode includes 'zip').

    Returns:
        (obs_path, pss_path, cog_path, date_part)

    Raises:
        ValueError, FileNotFoundError, NotADirectoryError, RuntimeError
    """
    source_path = Path(source_path)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    found = {}  # 'obs' / 'pss' / 'cog' → Path of the copied file

    def _copy(src: Path, match_name: str, is_zip: bool):
        """Match src against known patterns and copy it to the raw folder."""
        patterns = _ZIP_PATTERNS if is_zip else _FOLDER_PATTERNS
        for kind, pattern in patterns.items():
            if pattern.fullmatch(match_name):
                m = _DATE_RE.search(src.name)
                if not m:
                    print(f"  Warning: no date found in '{src.name}', skipping.")
                    return
                year, month, day = m.groups()
                dest_dir = raw_dir / f"Day_{year}{month}{day}"
                dest_dir.mkdir(exist_ok=True)
                dest = dest_dir / src.name
                shutil.copy2(src, dest)
                found[kind] = dest
                print(f"  Copied {src.name} → raw/{dest_dir.name}/")
                break

    if mode in ("zip", "both"):
        if not target_zip:
            raise ValueError("target_zip must be specified when mode includes 'zip'.")
        zip_path = source_path / target_zip
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            print(f"  Opening {target_zip} …")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for entry in sorted(zf.namelist()):
                    for pattern in _ZIP_PATTERNS.values():
                        if pattern.fullmatch(entry):
                            zf.extract(entry, tmp)
                            _copy(tmp / entry, entry, is_zip=True)
                            break

    if mode in ("folder", "both"):
        if not source_path.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {source_path}")
        for root, _, files in os.walk(source_path):
            for fname in files:
                _copy(Path(root) / fname, fname, is_zip=False)

    missing = [k for k in ('obs', 'pss', 'cog') if k not in found]
    if missing:
        raise RuntimeError(f"Could not find log file(s): {missing}")

    # Derive date_part from PSS filename: "PSS_YYYY_MM_DD_HH_MM_SS.csv" → "YYYY_MM_DD_HH_MM_SS"
    date_part = found['pss'].stem[4:]   # strip "PSS_" prefix

    return found['obs'], found['pss'], found['cog'], date_part


# ── Save ───────────────────────────────────────────────────────────────────────

def save_files(obs_df, bad_obs_df, pss_df, bad_pss_df, cog_df, bad_cog_df,
               header_lines, date_part, qc_dir, removed_dir):
    """
    Save QC and removed-lines CSVs into dated subfolders.

    Observer files get the original two-line header prepended.
    PSS and COG files are saved as plain CSVs.
    """
    qc_daily      = Path(qc_dir) / date_part
    removed_daily = Path(removed_dir) / date_part
    qc_daily.mkdir(parents=True, exist_ok=True)
    removed_daily.mkdir(parents=True, exist_ok=True)
    print(f"  QC folder:      {qc_daily}")
    print(f"  Removed folder: {removed_daily}")

    def _write_obs(df, path):
        with open(path, 'w', newline='') as f:
            f.write(header_lines[0].rstrip('\n') + '\n\n')
            df.to_csv(f, index=False)

    outputs = [
        (qc_daily      / f'ObserverLog_Detailed_QC_{date_part}.csv',           lambda p: _write_obs(obs_df, p)),
        (removed_daily / f'ObserverLog_Detailed_Removed_Lines_{date_part}.csv', lambda p: _write_obs(bad_obs_df, p)),
        (qc_daily      / f'PSS_QC_{date_part}.csv',                             lambda p: pss_df.to_csv(p, index=False)),
        (removed_daily / f'PSS_Removed_Lines_{date_part}.csv',                  lambda p: bad_pss_df.to_csv(p, index=False)),
        (qc_daily      / f'FinalCOG_QC_{date_part}.csv',                        lambda p: cog_df.to_csv(p, index=False)),
        (removed_daily / f'FinalCOG_Removed_Lines_{date_part}.csv',             lambda p: bad_cog_df.to_csv(p, index=False)),
    ]

    for path, writer in outputs:
        try:
            writer(path)
            print(f"  Saved {path.name}")
        except Exception as e:
            print(f"  ERROR saving {path.name}: {e}")
            raise


# ── Station corrections ────────────────────────────────────────────────────────

def apply_station_corrections(corrections, date_part, qc_dir, removed_dir, log_fn=None):
    """
    Write corrected station numbers back to the daily QC files, then rebuild
    the combined files.

    Args:
        corrections: dict of {str(file_num): float(new_station)}
        date_part:   subfolder name, e.g. '2026_04_14_06_50_12'
        qc_dir:      Path to QC_files/
        removed_dir: Path to lines_removed_files/
        log_fn:      optional callable for progress messages (defaults to print)

    Rules:
        OBS  Station              ← new_station (as-is)
        PSS  Station              ← new_station (as-is)
        COG  Source Point Station ← new_station − 0.5
    """
    log = log_fn or print
    qc_dir   = Path(qc_dir)
    daily_qc = qc_dir / date_part

    # Normalise keys: '2312.0' → '2312' so they match int columns in the CSVs
    corrections = {str(int(float(fn))): stn for fn, stn in corrections.items()}

    # ── OBS ──────────────────────────────────────────────────────────────────
    obs_path = daily_qc / f'ObserverLog_Detailed_QC_{date_part}.csv'
    with open(obs_path) as f:
        title_line = f.readline()
    obs_df = pd.read_csv(obs_path, skiprows=2)
    for fn, new_stn in corrections.items():
        obs_df.loc[obs_df['File#'].astype(str) == fn, 'Station'] = new_stn
    with open(obs_path, 'w', newline='') as f:
        f.write(title_line.rstrip('\n') + '\n\n')
        obs_df.to_csv(f, index=False)
    log(f"  Updated OBS: {obs_path.name}")

    # ── PSS ──────────────────────────────────────────────────────────────────
    pss_path = daily_qc / f'PSS_QC_{date_part}.csv'
    pss_df = pd.read_csv(pss_path)
    for fn, new_stn in corrections.items():
        pss_df.loc[pss_df['File Num'].astype(str) == fn, 'Station'] = new_stn
    pss_df.to_csv(pss_path, index=False)
    log(f"  Updated PSS: {pss_path.name}")

    # ── COG ──────────────────────────────────────────────────────────────────
    cog_path = daily_qc / f'FinalCOG_QC_{date_part}.csv'
    cog_df = pd.read_csv(cog_path)
    for fn, new_stn in corrections.items():
        cog_df.loc[cog_df['FF ID'].astype(str) == fn, 'Source Point Station'] = float(new_stn) - 0.5
    cog_df.to_csv(cog_path, index=False)
    log(f"  Updated COG: {cog_path.name}  (Source Point Station = station − 0.5)")

    # ── Rebuild combined ──────────────────────────────────────────────────────
    combine_files(qc_dir, Path(removed_dir))
    log("  Combined files rebuilt.")


# ── Combine ────────────────────────────────────────────────────────────────────

def combine_files(qc_dir, removed_dir):
    """
    Merge all daily CSVs of each type into a single combined file.

    Files are sorted by their key column (File#, File Num, FF ID) and deduplicated.
    """
    qc_dir       = Path(qc_dir)
    removed_dir  = Path(removed_dir)
    combined_qc      = qc_dir / 'combined'
    combined_removed = removed_dir / 'combined'
    combined_qc.mkdir(exist_ok=True)
    combined_removed.mkdir(exist_ok=True)

    # (glob_pattern, output_name, base_folder, out_folder, obs_header, sort_column)
    file_types = [
        ('ObserverLog_Detailed_QC_*.csv',            'ObserverLog_Detailed_QC_Combined.csv',            qc_dir,      combined_qc,      True,  'File#'),
        ('ObserverLog_Detailed_Removed_Lines_*.csv', 'ObserverLog_Detailed_Removed_Lines_Combined.csv', removed_dir, combined_removed, True,  None),
        ('PSS_QC_*.csv',                             'PSS_QC_Combined.csv',                             qc_dir,      combined_qc,      False, 'File Num'),
        ('PSS_Removed_Lines_*.csv',                  'PSS_Removed_Lines_Combined.csv',                  removed_dir, combined_removed, False, None),
        ('FinalCOG_QC_*.csv',                        'FinalCOG_QC_Combined.csv',                        qc_dir,      combined_qc,      False, 'FF ID'),
        ('FinalCOG_Removed_Lines_*.csv',             'FinalCOG_Removed_Lines_Combined.csv',             removed_dir, combined_removed, False, None),
    ]

    for pattern, out_name, base, out_folder, obs_header, sort_col in file_types:
        out_path = out_folder / out_name
        if out_path.exists():
            out_path.unlink()

        # Collect files from dated subfolders, skipping the 'combined' folder itself
        daily_files = []
        for daily in sorted(base.iterdir()):
            if daily.is_dir() and daily.name.lower() != 'combined':
                daily_files.extend(glob.glob(str(daily / pattern)))
        daily_files.sort(key=lambda x: Path(x).stem.split('_')[-1])

        if not daily_files:
            print(f"  No files found for {pattern}")
            continue

        frames = []
        seen_dates = set()
        last_header = None

        for fp in daily_files:
            date_key = Path(fp).stem.split('_')[-1]
            if date_key in seen_dates:
                print(f"  Skipping duplicate: {Path(fp).name}")
                continue
            try:
                if obs_header:
                    with open(fp) as f:
                        lines = f.readlines()
                    if not lines:
                        continue
                    last_header = lines[0]
                    df = pd.read_csv(fp, skiprows=2)
                else:
                    df = pd.read_csv(fp)
                frames.append(df)
                seen_dates.add(date_key)
            except Exception as e:
                print(f"  Error reading {Path(fp).name}: {e}")

        if not frames:
            print(f"  No valid data for {out_name}")
            continue

        final = pd.concat(frames, ignore_index=True).drop_duplicates()

        # Sort by the relevant key column
        if sort_col and sort_col in final.columns:
            final = final.sort_values(sort_col).reset_index(drop=True)
        elif 'TB Local Time' in final.columns:
            final['TB Local Time'] = pd.to_datetime(
                final['TB Local Time'], format='%Y/%m/%d %H:%M:%S.%f', errors='coerce'
            )
            final = final.sort_values('TB Local Time').reset_index(drop=True)
        elif 'Date' in final.columns:
            final['Date'] = pd.to_datetime(final['Date'], errors='coerce')
            final = final.sort_values('Date').reset_index(drop=True)

        if obs_header and last_header:
            with open(out_path, 'w', newline='') as f:
                f.write(last_header.rstrip('\n') + '\n\n')
                final.to_csv(f, index=False)
        else:
            final.to_csv(out_path, index=False)

        print(f"  Combined {len(frames)} file(s) → {out_name}")
