import pandas as pd


# ── Observer Log ───────────────────────────────────────────────────────────────

def process_observer_log(file_path):
    """
    Process the Observer Log CSV.

    Removes:
    - Rows where Status == 'Void'
    - Rows where PSS Info shows '0 of N PSS Received'
    - Rows with blank or non-numeric File# or Station

    Returns:
        (obs_df, bad_obs_df, header_lines, removal_reasons)
    """
    with open(file_path, 'r') as f:
        header_lines = [next(f) for _ in range(2)]

    df = pd.read_csv(file_path, header=2)
    bad_df = pd.DataFrame(columns=df.columns)
    reasons = {}

    def flag(file_num, reason):
        reasons.setdefault(file_num, []).append(reason)

    # Split Void rows: those with a non-void replacement are removed immediately;
    # those with no replacement are returned as void_candidates for interactive review.
    void_mask      = df['Status'] == 'Void'
    void_with_repl = []
    void_no_repl   = []

    for idx in df[void_mask].index:
        file_num = df.loc[idx, 'File#']
        line, station = df.loc[idx, 'Line'], df.loc[idx, 'Station']
        flag(file_num, 'Status is Void')
        has_replacement = (
            (df['Line']    == line)    &
            (df['Station'] == station) &
            (df.index      != idx)     &
            (df['Status']  != 'Void')
        ).any()
        if has_replacement:
            flag(file_num, f'Replacement found for Line={line}, Station={station}')
            void_with_repl.append(idx)
        else:
            flag(file_num, f'No replacement found for Line={line}, Station={station}')
            void_no_repl.append(idx)

    # Capture void_candidates BEFORE any drop so original indices are still valid
    void_candidates = df.loc[void_no_repl].copy() if void_no_repl else pd.DataFrame(columns=df.columns)

    # Remove all void rows from df in one pass (avoids index-invalidation after reset)
    if void_with_repl:
        bad_df = pd.concat([bad_df, df.loc[void_with_repl]], ignore_index=True)
    all_void_indices = void_with_repl + void_no_repl
    if all_void_indices:
        df = df.drop(index=all_void_indices).reset_index(drop=True)

    # Remove rows with 0 PSS received
    pss_zero_mask = df[' PSS Info'].str.match(r'0 of \d+ PSS Received', na=False)
    for idx in df[pss_zero_mask].index:
        flag(df.loc[idx, 'File#'], 'PSS Info is 0 of * PSS Received')
    bad_df = pd.concat([bad_df, df[pss_zero_mask]], ignore_index=True)
    df = df[~pss_zero_mask].reset_index(drop=True)

    # Remove rows with blank or non-numeric File# / Station
    bad_indices = set()
    for col in ['File#', 'Station']:
        empty_mask = df[col].isna() | (df[col] == '')
        for idx in df[empty_mask].index:
            flag(df.loc[idx, 'File#'], f'{col} is empty or NaN')
        bad_indices.update(df[empty_mask].index)

    for col, label in [('File#', 'File# is non-numeric'), ('Station', 'Station is non-numeric')]:
        non_numeric = ~pd.to_numeric(df[col], errors='coerce').notna()
        for idx in df[non_numeric].index:
            flag(df.loc[idx, 'File#'], label)
        bad_indices.update(df[non_numeric].index)

    if bad_indices:
        bad_df = pd.concat([bad_df, df.loc[list(bad_indices)]], ignore_index=True)
        df = df.drop(index=list(bad_indices)).reset_index(drop=True)

    return df, bad_df, void_candidates, header_lines, reasons


# ── PSS Log ────────────────────────────────────────────────────────────────────

def process_pss_log(file_path):
    """
    Process the PSS CSV.

    Removes:
    - Rows where Void == 'Void'
    - Rows with a blank File Num
    - Rows with a blank or zero Sweep Checksum

    Returns:
        (pss_df, bad_pss_df, removal_reasons)
    """
    df = pd.read_csv(file_path, header=0)
    bad_df = pd.DataFrame(columns=df.columns)
    reasons = {}

    def flag(file_num, idx, reason):
        key = f"File Num {file_num}" if pd.notna(file_num) else f"Index {idx} (no File Num)"
        reasons.setdefault(key, []).append(reason)

    # Remove Void rows
    void_mask = df['Void'] == 'Void'
    for idx in df[void_mask].index:
        flag(df.loc[idx, 'File Num'], idx, 'Void entry')
    bad_df = pd.concat([bad_df, df[void_mask]], ignore_index=True)
    df = df[~void_mask].reset_index(drop=True)

    # Remove blank File Num
    blank_filenum = df['File Num'].isna() | (df['File Num'] == '')
    for idx in df[blank_filenum].index:
        flag(None, idx, 'Blank File Num')
    bad_df = pd.concat([bad_df, df[blank_filenum]], ignore_index=True)
    df = df[~blank_filenum].reset_index(drop=True)

    # Remove blank or zero Sweep Checksum
    df['Sweep Checksum'] = df['Sweep Checksum'].astype(str).str.strip()
    invalid_checksum = df['Sweep Checksum'].isin(['', 'nan', '0', '0.0'])
    for idx in df[invalid_checksum].index:
        flag(df.loc[idx, 'File Num'], idx, 'Blank or zero Sweep Checksum')
    bad_df = pd.concat([bad_df, df[invalid_checksum]], ignore_index=True)
    df = df[~invalid_checksum].reset_index(drop=True)

    # Deduplicate by (File Num, Unit ID) — keep the last sweep per vibe per shot.
    # Multiple rows can occur when a shot is retried (same File Num re-recorded);
    # counting all of them inflates GPS quality tallies and PSS entry checks.
    dup_mask = df.duplicated(subset=['File Num', 'Unit ID'], keep='last')
    for idx in df[dup_mask].index:
        flag(df.loc[idx, 'File Num'], idx, 'Duplicate (File Num, Unit ID) — earlier sweep discarded')
    bad_df = pd.concat([bad_df, df[dup_mask]], ignore_index=True)
    df = df[~dup_mask].reset_index(drop=True)

    return df, bad_df, reasons


# ── COG Log ────────────────────────────────────────────────────────────────────

def process_cog_log(file_path):
    """
    Load the FinalCOG CSV with no filtering (stub for future checks).

    Returns:
        (cog_df, bad_cog_df, removal_reasons)
    """
    df = pd.read_csv(file_path, header=0)
    return df, pd.DataFrame(columns=df.columns), {}


# ── Summary & Validation ───────────────────────────────────────────────────────

def print_summary(obs_df, bad_obs_df, pss_df, bad_pss_df, cog_df, bad_cog_df,
                  obs_reasons, pss_reasons, cog_reasons):
    """Print a processing summary for all three log files."""
    sep = '─' * 60

    print(sep)
    print("Observer Log")
    if obs_reasons:
        for file_num, r in sorted(obs_reasons.items()):
            print(f"  Removed File# {file_num}: {'; '.join(r)}")
    else:
        print("  No rows removed.")
    print(f"  Kept: {len(obs_df)} rows  |  Removed: {len(bad_obs_df)} rows")

    print(sep)
    print("PSS Log")
    if pss_reasons:
        for key, r in sorted(pss_reasons.items()):
            print(f"  Removed {key}: {'; '.join(set(r))}")
    else:
        print("  No rows removed.")
    print(f"  Kept: {len(pss_df)} rows ({len(pss_df.columns)} columns)  |  Removed: {len(bad_pss_df)} rows")

    print(sep)
    print("FinalCOG Log")
    print(f"  Kept: {len(cog_df)} rows ({len(cog_df.columns)} columns)  |  Removed: {len(bad_cog_df)} rows")
    if cog_reasons:
        print(f"  Removal reasons: {cog_reasons}")
    print(sep)


def compare_entries(obs_df, pss_df, cog_df, expected_pss_per_point):
    """Cross-validate file numbers across all three log files."""
    if 'File#' not in obs_df or 'File Num' not in pss_df or 'FF ID' not in cog_df:
        print("  ERROR: Required columns missing in one or more DataFrames.")
        return

    # Check PSS entry counts per shot point
    pss_counts = pss_df['File Num'].astype(str).value_counts()
    wrong_counts = pss_counts[pss_counts != expected_pss_per_point]

    print(f"\n  PSS entry count check (expected {expected_pss_per_point} per shot point):")
    if wrong_counts.empty:
        print("    All File Num entries have the expected count.")
    else:
        for file_num, count in wrong_counts.items():
            print(f"    File Num {file_num}: {count} entries (expected {expected_pss_per_point})")

    # Cross-file comparison
    obs_nums = set(obs_df['File#'].astype(str))
    pss_nums = set(pss_df['File Num'].astype(str))
    cog_nums = set(cog_df['FF ID'].astype(str))
    all_nums = obs_nums | pss_nums | cog_nums

    print(f"\n  Unique file numbers — OBS: {len(obs_nums)}  PSS: {len(pss_nums)}  COG: {len(cog_nums)}  Total: {len(all_nums)}")

    obs_missing = all_nums - obs_nums
    pss_missing = all_nums - pss_nums
    cog_missing = all_nums - cog_nums

    if any([obs_missing, pss_missing, cog_missing]):
        if obs_missing:
            print(f"    Missing from OBS: {sorted(obs_missing)}")
        if pss_missing:
            print(f"    Missing from PSS: {sorted(pss_missing)}")
        if cog_missing:
            print(f"    Missing from COG: {sorted(cog_missing)}")
    else:
        print("    All file numbers present in all three logs.")
