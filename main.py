from pathlib import Path

from processors import (
    process_observer_log,
    process_pss_log,
    process_cog_log,
    print_summary,
    compare_entries,
)
from file_io import gather_files, save_files, combine_files

# ── Input ──────────────────────────────────────────────────────────────────────
SOURCE_DIR      = r"D:\vcloud\Anglo_Planet_Downs_2D_2025_Job1071\End of Day Output"
TARGET_ZIP      = "09_13_2025_15_53_44.zip"
MODE            = "zip"         # "zip", "folder", or "both"
VIBES_PER_POINT = 1             # Expected number of PSS entries per shot point

# ── Output ─────────────────────────────────────────────────────────────────────
RAW_DIR      = Path(r"C:\Users\jstep\OneDrive\Desktop\obs_checker_regal\raw")
QC_DIR       = Path(r"C:\Users\jstep\OneDrive\Desktop\obs_checker_regal\QC_files")
REMOVED_DIR  = Path(r"C:\Users\jstep\OneDrive\Desktop\obs_checker_regal\lines_removed_files")

# ───────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══════════════════════════════════════════")
    print("       Seismic Log QC Checker")
    print("══════════════════════════════════════════")

    QC_DIR.mkdir(parents=True, exist_ok=True)
    REMOVED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Gather files
    print("\n[1/6] Gathering files …")
    obs_file, pss_file, cog_file, date_part = gather_files(
        SOURCE_DIR, RAW_DIR, mode=MODE, target_zip=TARGET_ZIP
    )

    # 2. Process each log
    print("\n[2/6] Processing logs …")
    obs_df, bad_obs_df, header_lines, obs_reasons = process_observer_log(obs_file)
    pss_df, bad_pss_df, pss_reasons              = process_pss_log(pss_file)
    cog_df, bad_cog_df, cog_reasons              = process_cog_log(cog_file)

    # 3. Print processing summary
    print("\n[3/6] Processing summary")
    print_summary(
        obs_df, bad_obs_df, pss_df, bad_pss_df, cog_df, bad_cog_df,
        obs_reasons, pss_reasons, cog_reasons,
    )

    # 4. Cross-file validation
    print("\n[4/6] Cross-file validation")
    compare_entries(obs_df, pss_df, cog_df, VIBES_PER_POINT)

    # 5. Save QC output
    print("\n[5/6] Saving QC files …")
    save_files(
        obs_df, bad_obs_df, pss_df, bad_pss_df, cog_df, bad_cog_df,
        header_lines, date_part, QC_DIR, REMOVED_DIR,
    )

    # 6. Combine daily files
    print("\n[6/6] Combining daily files …")
    combine_files(QC_DIR, REMOVED_DIR)

    print("\n══════════════════════════════════════════")
    print("  Done.")
    print("══════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
