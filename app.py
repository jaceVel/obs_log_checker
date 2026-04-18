import sys
import threading
from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PyQt5.QtCore import QSettings, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

SCRIPT_DIR  = Path(__file__).parent
RAW_DIR     = SCRIPT_DIR / "raw"
QC_DIR      = SCRIPT_DIR / "QC_files"
REMOVED_DIR = SCRIPT_DIR / "lines_removed_files"

_PSS_METRICS = ['Phase Max', 'Phase Avg', 'Force Max', 'Force Avg', 'THD Max', 'THD Avg']


# ── Stdout redirector ──────────────────────────────────────────────────────────

class _Stream:
    def __init__(self, signal):
        self._signal = signal

    def write(self, text):
        if text and text.strip():
            self._signal.emit(text)

    def flush(self):
        pass


# ── Duplicate comparison data builder ─────────────────────────────────────────

def _build_comparison_data(obs_df, pss_df, cog_df):
    """Return list of duplicate-group dicts for the review dialog."""
    dup_mask = obs_df.duplicated(subset=['Line', 'Station'], keep=False)
    dups     = obs_df[dup_mask]
    if dups.empty:
        return []

    # PSS lookup: File Num → aggregated metric strings
    pss_lookup = {}
    if 'File Num' in pss_df.columns:
        for file_num, grp in pss_df.groupby('File Num'):
            row = {}
            for col in _PSS_METRICS:
                if col in grp.columns:
                    vals = pd.to_numeric(grp[col], errors='coerce').dropna()
                    if not vals.empty:
                        row[col] = f"{vals.max():.2f}" if col.endswith('Max') else f"{vals.mean():.2f}"
                    else:
                        row[col] = 'N/A'
                else:
                    row[col] = 'N/A'
            pss_lookup[str(file_num)] = row

    # COG lookup: FF ID → Distance to Source Point
    cog_lookup = {}
    if 'FF ID' in cog_df.columns and 'Distance to Source Point' in cog_df.columns:
        for _, row in cog_df.iterrows():
            cog_lookup[str(row['FF ID'])] = str(row['Distance to Source Point'])

    groups = []
    for (line, station), grp in dups.groupby(['Line', 'Station']):
        shots = []
        for _, obs_row in grp.iterrows():
            fn  = str(obs_row['File#'])
            pss = pss_lookup.get(fn, {})
            shots.append({
                'file_num':  fn,
                'pss_info':  str(obs_row.get(' PSS Info', 'N/A')).strip(),
                'phase_max': pss.get('Phase Max', 'N/A'),
                'phase_avg': pss.get('Phase Avg', 'N/A'),
                'force_max': pss.get('Force Max', 'N/A'),
                'force_avg': pss.get('Force Avg', 'N/A'),
                'thd_max':   pss.get('THD Max', 'N/A'),
                'thd_avg':   pss.get('THD Avg', 'N/A'),
                'distance':  cog_lookup.get(fn, 'N/A'),
            })
        groups.append({'line': str(line), 'station': str(station), 'shots': shots})

    return groups


# ── Void candidate data builder ───────────────────────────────────────────────

def _build_void_data(void_candidates, pss_df, bad_pss_df, cog_df):
    """Return list of shot dicts for the void review dialog."""
    if void_candidates.empty:
        return []

    # Combine kept and removed PSS rows — void shots' PSS entries were filtered out
    all_pss = pd.concat([pss_df, bad_pss_df], ignore_index=True)

    # PSS lookup: File Num → aggregated metric strings
    pss_lookup = {}
    if 'File Num' in all_pss.columns:
        for file_num, grp in all_pss.groupby('File Num'):
            row = {}
            for col in _PSS_METRICS:
                if col in grp.columns:
                    vals = pd.to_numeric(grp[col], errors='coerce').dropna()
                    if not vals.empty:
                        row[col] = f"{vals.max():.2f}" if col.endswith('Max') else f"{vals.mean():.2f}"
                    else:
                        row[col] = 'N/A'
                else:
                    row[col] = 'N/A'
            pss_lookup[str(file_num)] = row

    # COG lookup: FF ID → Distance to Source Point
    cog_lookup = {}
    if 'FF ID' in cog_df.columns and 'Distance to Source Point' in cog_df.columns:
        for _, row in cog_df.iterrows():
            cog_lookup[str(row['FF ID'])] = str(row['Distance to Source Point'])

    shots = []
    for _, obs_row in void_candidates.iterrows():
        fn  = str(obs_row['File#'])
        pss = pss_lookup.get(fn, {})
        shots.append({
            'file_num':  fn,
            'line':      str(obs_row['Line']),
            'station':   str(obs_row['Station']),
            'pss_info':  str(obs_row.get(' PSS Info', 'N/A')).strip(),
            'phase_max': pss.get('Phase Max', 'N/A'),
            'phase_avg': pss.get('Phase Avg', 'N/A'),
            'force_max': pss.get('Force Max', 'N/A'),
            'force_avg': pss.get('Force Avg', 'N/A'),
            'thd_max':   pss.get('THD Max', 'N/A'),
            'thd_avg':   pss.get('THD Avg', 'N/A'),
            'distance':  cog_lookup.get(fn, 'N/A'),
        })

    return shots


# ── Synthetic COG row builder ─────────────────────────────────────────────────

def _build_synthetic_cog_row(obs_row, pss_rows, cog_columns):
    """
    Reconstruct a COG entry for a reinstated void shot using OBS + PSS data.

    Decoder Lat/Lon  = mean of all vibe Lat/Lon values (exact for 1 or 2 vibes).
    Decoder Elevation = round(mean(Altitude), 1)  — matches COG system behaviour.
    Decoder X / Y    = Lon / Lat  (COG convention).
    Source Point geometry and Near Flag fields are left as NaN (geometry file needed).
    """
    decoder_lat = pss_rows['Lat'].mean()
    decoder_lon = pss_rows['Lon'].mean()
    decoder_elv = round(float(pss_rows['Altitude'].mean()), 1)
    first       = pss_rows.iloc[0]

    row = {col: pd.NA for col in cog_columns}
    row['FF ID']                       = first['File Num']
    row['VP ID']                       = first['File Num']
    row['EP']                          = first.get('EP ID',         pd.NA)
    row['Encoder Index']               = first.get('Encoder Index', pd.NA)
    row['Group ID']                    = obs_row.get('Source Group', pd.NA)
    row['Decoder Lat']                 = decoder_lat
    row['Decoder Lon']                 = decoder_lon
    row['Decoder X']                   = decoder_lon
    row['Decoder Y']                   = decoder_lat
    row['Decoder Elevation']           = decoder_elv
    row['Source Point Line']           = obs_row.get('Line',    pd.NA)
    raw_stn = obs_row.get('Station', pd.NA)
    row['Source Point Station'] = int(float(raw_stn)) if pd.notna(raw_stn) else pd.NA
    row['UTC Time']                    = first.get('TB UTC Time',      pd.NA)
    row['Local Time']                  = first.get('TB Local Time',    pd.NA)
    row['GPS Quality']                 = 0
    row['Start Time Delta']            = first.get('Start Time Delta', pd.NA)
    row['Unit']                        = 'ARC Degrees'
    row['PSS Info']                    = obs_row.get(' PSS Info', pd.NA)
    row['Source Type']                 = obs_row.get('SRC Type',  pd.NA)
    # Encoder GPS fields — 0 to match observed COG pattern (encoder-system GPS, not vibe GPS)
    row['Sats'] = row['PDOP'] = row['HDOP'] = row['VDOP'] = row['Age'] = 0
    return row


# ── Worker thread ──────────────────────────────────────────────────────────────

class Worker(QThread):
    log              = pyqtSignal(str)
    voids_found      = pyqtSignal(list)   # no-replacement void shots for review
    duplicates_found = pyqtSignal(list)
    results_ready    = pyqtSignal(object, object, object, str)   # obs_df, cog_df, pss_df, date_part
    done             = pyqtSignal(bool, str)

    def __init__(self, source_dir, target_zip, mode, vibes_per_point):
        super().__init__()
        self.source_dir           = source_dir
        self.target_zip           = target_zip
        self.mode                 = mode
        self.vibes_per_point      = vibes_per_point
        self._void_resume_event   = threading.Event()
        self._void_reinstate      = set()
        self._resume_event        = threading.Event()
        self._selections          = {}

    def set_void_selections(self, reinstate_set):
        self._void_reinstate = reinstate_set
        self._void_resume_event.set()

    def set_selections(self, selections):
        self._selections = selections
        self._resume_event.set()

    def run(self):
        old_stdout = sys.stdout
        sys.stdout = _Stream(self.log)
        try:
            from processors import (
                compare_entries,
                print_summary,
                process_cog_log,
                process_observer_log,
                process_pss_log,
            )
            from file_io import combine_files, gather_files, save_files

            QC_DIR.mkdir(parents=True, exist_ok=True)
            REMOVED_DIR.mkdir(parents=True, exist_ok=True)

            print("\n══════════════════════════════════════════")
            print("       Seismic Log QC Checker")
            print("══════════════════════════════════════════")

            print("\n[1/6] Gathering files …")
            obs_file, pss_file, cog_file, date_part = gather_files(
                self.source_dir, RAW_DIR,
                mode=self.mode, target_zip=self.target_zip,
            )

            print("\n[2/6] Processing logs …")
            obs_df, bad_obs_df, void_candidates, header_lines, obs_reasons = process_observer_log(obs_file)
            pss_df, bad_pss_df, pss_reasons                                = process_pss_log(pss_file)
            cog_df, bad_cog_df, cog_reasons                                = process_cog_log(cog_file)

            # ── Void review ──────────────────────────────────────────────────
            if not void_candidates.empty:
                n = len(void_candidates)
                print(f"\n  Found {n} void shot{'s' if n > 1 else ''} with no replacement — waiting for review …")
                void_data = _build_void_data(void_candidates, pss_df, bad_pss_df, cog_df)
                self.voids_found.emit(void_data)
                self._void_resume_event.wait()
                obs_df, bad_obs_df, obs_reasons, \
                pss_df, bad_pss_df, pss_reasons, \
                cog_df, bad_cog_df = self._apply_void_selections(
                    obs_df, bad_obs_df, obs_reasons,
                    pss_df, bad_pss_df, pss_reasons,
                    cog_df, bad_cog_df,
                    void_candidates,
                )
            else:
                print("\n  No unmatched void shots found.")

            # ── Duplicate check ──────────────────────────────────────────────
            dup_groups = _build_comparison_data(obs_df, pss_df, cog_df)
            if dup_groups:
                n = len(dup_groups)
                print(f"\n  Found {n} duplicate shot group{'s' if n > 1 else ''} — waiting for review …")
                self.duplicates_found.emit(dup_groups)
                self._resume_event.wait()
                obs_df, bad_obs_df, obs_reasons, \
                pss_df, bad_pss_df, pss_reasons, \
                cog_df, bad_cog_df = self._apply_selections(
                    obs_df, bad_obs_df, obs_reasons,
                    pss_df, bad_pss_df, pss_reasons,
                    cog_df, bad_cog_df,
                )
            else:
                print("\n  No duplicate shots found.")

            # Emit QC'd data for the Stn Num Check and Vibe QC tabs
            _cog_pos_cols = {'FF ID', 'Decoder Lat', 'Decoder Lon',
                             'Distance to Source Point',
                             'Source Point Lat', 'Source Point Lon'}
            self.results_ready.emit(
                obs_df[['File#', 'Line', 'Station']].copy(),
                cog_df[list(_cog_pos_cols)].copy()
                if _cog_pos_cols.issubset(cog_df.columns)
                else pd.DataFrame(),
                pss_df.copy(),
                date_part,
            )

            # Normalise reason-dict keys so sorted() works regardless of dtype
            obs_reasons = {str(k): v for k, v in obs_reasons.items()}
            pss_reasons = {str(k): v for k, v in pss_reasons.items()}
            cog_reasons = {str(k): v for k, v in cog_reasons.items()}

            print("\n[3/6] Processing summary")
            print_summary(
                obs_df, bad_obs_df, pss_df, bad_pss_df, cog_df, bad_cog_df,
                obs_reasons, pss_reasons, cog_reasons,
            )

            print("\n[4/6] Cross-file validation")
            compare_entries(obs_df, pss_df, cog_df, self.vibes_per_point)

            print("\n[5/6] Saving QC files …")
            save_files(
                obs_df, bad_obs_df, pss_df, bad_pss_df, cog_df, bad_cog_df,
                header_lines, date_part, QC_DIR, REMOVED_DIR,
            )

            print("\n[6/6] Combining daily files …")
            combine_files(QC_DIR, REMOVED_DIR)

            print("\n══════════════════════════════════════════")
            print("  Done.")
            print("══════════════════════════════════════════\n")

            self.done.emit(True, "QC complete.")

        except Exception as exc:
            print(f"\nERROR: {exc}")
            self.done.emit(False, str(exc))

        finally:
            sys.stdout = old_stdout

    def _apply_void_selections(self, obs_df, bad_obs_df, obs_reasons,
                               pss_df, bad_pss_df, pss_reasons,
                               cog_df, bad_cog_df,
                               void_candidates):
        import pandas as pd
        for _, row in void_candidates.iterrows():
            fn     = str(row['File#'])
            fn_num = pd.to_numeric(fn, errors='coerce')

            if fn in self._void_reinstate:
                # ── OBS: reinstate with Status = Acquired ─────────────────────
                new_row = row.copy()
                new_row['Status'] = 'Acquired'
                obs_df = pd.concat([obs_df, new_row.to_frame().T], ignore_index=True)
                obs_keys = [k for k in obs_reasons if str(k) == fn]
                for k in obs_keys:
                    del obs_reasons[k]

                # ── PSS: move matching rows back, clear Void column ───────────
                pss_mask     = pd.to_numeric(bad_pss_df['File Num'], errors='coerce') == fn_num
                pss_shot_rows = bad_pss_df[pss_mask].copy()
                if pss_mask.any():
                    pss_shot_rows['Void'] = ''
                    pss_df     = pd.concat([pss_df, pss_shot_rows], ignore_index=True)
                    bad_pss_df = bad_pss_df[~pss_mask].reset_index(drop=True)
                    pss_keys   = [k for k in pss_reasons if fn in str(k)]
                    for k in pss_keys:
                        del pss_reasons[k]

                # ── COG: synthesise entry from OBS + PSS decoder positions ─────
                if not pss_shot_rows.empty:
                    syn_row = _build_synthetic_cog_row(row, pss_shot_rows, cog_df.columns)
                    cog_df  = pd.concat([cog_df, pd.DataFrame([syn_row])], ignore_index=True)
                    print(f"  Built synthetic COG entry for File# {fn} "
                          f"({len(pss_shot_rows)} vibe(s), "
                          f"Lat={syn_row['Decoder Lat']:.6f}, "
                          f"Lon={syn_row['Decoder Lon']:.6f})")
                else:
                    print(f"  WARNING: no PSS rows found for File# {fn} — COG entry skipped")

                print(f"  Reinstated void File# {fn} "
                      f"(Line {row['Line']}, Station {row['Station']})")
            else:
                bad_obs_df = pd.concat([bad_obs_df, row.to_frame().T], ignore_index=True)

        # Re-sort all three files so reinstated rows sit in their correct positions
        obs_df = (
            obs_df
            .assign(_sort=pd.to_numeric(obs_df['File#'], errors='coerce'))
            .sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
        )
        pss_df = (
            pss_df
            .assign(_sort=pd.to_numeric(pss_df['File Num'], errors='coerce'))
            .sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
        )
        cog_df = (
            cog_df
            .assign(_sort=pd.to_numeric(cog_df['FF ID'], errors='coerce'))
            .sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
        )
        return obs_df, bad_obs_df, obs_reasons, pss_df, bad_pss_df, pss_reasons, cog_df, bad_cog_df

    def _apply_selections(self, obs_df, bad_obs_df, obs_reasons,
                          pss_df, bad_pss_df, pss_reasons,
                          cog_df, bad_cog_df):
        for (line, station), keep_num in self._selections.items():
            discard_mask = (
                (obs_df['Line'].astype(str)    == str(line))    &
                (obs_df['Station'].astype(str) == str(station)) &
                (obs_df['File#'].astype(str)   != str(keep_num))
            )
            discarded = obs_df[discard_mask]

            for _, row in discarded.iterrows():
                fn = str(row['File#'])
                obs_reasons.setdefault(fn, []).append(
                    f'Duplicate shot — kept File# {keep_num}'
                )
                print(f"  Discarding duplicate File# {fn} "
                      f"(Line {line}, Station {station}) — kept {keep_num}")

                if 'File Num' in pss_df.columns:
                    pss_mask   = pss_df['File Num'].astype(str) == fn
                    bad_pss_df = pd.concat([bad_pss_df, pss_df[pss_mask]], ignore_index=True)
                    pss_df     = pss_df[~pss_mask].reset_index(drop=True)
                    pss_reasons.setdefault(f'File Num {fn}', []).append(
                        f'Duplicate shot — kept File# {keep_num}'
                    )

                if 'FF ID' in cog_df.columns:
                    cog_mask   = cog_df['FF ID'].astype(str) == fn
                    bad_cog_df = pd.concat([bad_cog_df, cog_df[cog_mask]], ignore_index=True)
                    cog_df     = cog_df[~cog_mask].reset_index(drop=True)

            bad_obs_df = pd.concat([bad_obs_df, discarded], ignore_index=True)
            obs_df     = obs_df[~discard_mask].reset_index(drop=True)

        return (obs_df, bad_obs_df, obs_reasons,
                pss_df, bad_pss_df, pss_reasons,
                cog_df, bad_cog_df)


# ── Void review dialog ────────────────────────────────────────────────────────

class VoidReviewDialog(QDialog):
    _COLS = [
        ("File#",       'file_num'),
        ("Line",        'line'),
        ("Station",     'station'),
        ("PSS Info",    'pss_info'),
        ("Phase Max",   'phase_max'),
        ("Phase Avg",   'phase_avg'),
        ("Force Max",   'force_max'),
        ("Force Avg",   'force_avg'),
        ("THD Max",     'thd_max'),
        ("THD Avg",     'thd_avg'),
        ("Dist. to SP", 'distance'),
    ]

    def __init__(self, shots, parent=None):
        super().__init__(parent)
        self._shots      = shots
        self._checkboxes = []
        self.setWindowTitle("Void Shot Review — No Replacement Found")
        self.setMinimumSize(1020, 380)
        self.setModal(True)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        n    = len(self._shots)
        info = QLabel(
            f"<b>Found {n} void shot{'s' if n > 1 else ''} with no replacement recorded.</b>  "
            f"Tick any that were voided by mistake and should be reinstated, then click Confirm."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        n_cols = len(self._COLS) + 1
        table  = QTableWidget(n, n_cols)
        table.setHorizontalHeaderLabels(["Reinstate"] + [c[0] for c in self._COLS])
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.horizontalHeader().setStretchLastSection(False)

        for row_i, shot in enumerate(self._shots):
            cb_wrap = QWidget()
            cb_hbox = QHBoxLayout(cb_wrap)
            cb_hbox.setContentsMargins(4, 0, 4, 0)
            cb_hbox.setAlignment(Qt.AlignCenter)
            cb = QCheckBox()
            self._checkboxes.append(cb)
            cb_hbox.addWidget(cb)
            table.setCellWidget(row_i, 0, cb_wrap)

            for col_i, (_, key) in enumerate(self._COLS):
                item = QTableWidgetItem(str(shot.get(key, 'N/A')))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_i, col_i + 1, item)

        table.resizeColumnsToContents()
        header_h = table.horizontalHeader().height()
        rows_h   = sum(table.rowHeight(i) for i in range(n))
        table.setFixedHeight(min(header_h + rows_h + 4, 520))
        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                   Qt.Horizontal, self)
        buttons.button(QDialogButtonBox.Ok).setText("Confirm")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_reinstate_file_nums(self):
        return {
            self._shots[i]['file_num']
            for i, cb in enumerate(self._checkboxes)
            if cb.isChecked()
        }


# ── Duplicate review dialog ────────────────────────────────────────────────────

class DuplicateReviewDialog(QDialog):
    _COLS = [
        ("File#",       'file_num'),
        ("PSS Info",    'pss_info'),
        ("Phase Max",   'phase_max'),
        ("Phase Avg",   'phase_avg'),
        ("Force Max",   'force_max'),
        ("Force Avg",   'force_avg'),
        ("THD Max",     'thd_max'),
        ("THD Avg",     'thd_avg'),
        ("Dist. to SP", 'distance'),
    ]

    def __init__(self, groups, parent=None):
        super().__init__(parent)
        self._groups     = groups
        self._btn_groups = []
        self.setWindowTitle("Duplicate Shot Review")
        self.setMinimumSize(960, 520)
        self.setModal(True)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        n    = len(self._groups)
        info = QLabel(
            f"<b>Found {n} duplicate shot group{'s' if n > 1 else ''}.</b>  "
            f"Select which shot to keep for each group, then click Confirm."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        scroll      = QScrollArea()
        scroll.setWidgetResizable(True)
        container   = QWidget()
        cont_layout = QVBoxLayout(container)
        cont_layout.setSpacing(12)

        for group in self._groups:
            gb        = QGroupBox(f"Line: {group['line']}   ·   Station: {group['station']}")
            gb_layout = QVBoxLayout(gb)
            n_shots   = len(group['shots'])
            n_cols    = len(self._COLS) + 1

            table = QTableWidget(n_shots, n_cols)
            table.setHorizontalHeaderLabels(["Keep"] + [c[0] for c in self._COLS])
            table.verticalHeader().setVisible(False)
            table.setSelectionMode(QTableWidget.NoSelection)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.horizontalHeader().setStretchLastSection(False)

            btn_group = QButtonGroup(self)
            self._btn_groups.append((group, btn_group))

            for row_i, shot in enumerate(group['shots']):
                radio_wrap = QWidget()
                radio_hbox = QHBoxLayout(radio_wrap)
                radio_hbox.setContentsMargins(4, 0, 4, 0)
                radio_hbox.setAlignment(Qt.AlignCenter)
                radio = QRadioButton()
                btn_group.addButton(radio, row_i)
                radio_hbox.addWidget(radio)
                table.setCellWidget(row_i, 0, radio_wrap)

                for col_i, (_, key) in enumerate(self._COLS):
                    item = QTableWidgetItem(str(shot.get(key, 'N/A')))
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(row_i, col_i + 1, item)

            table.resizeColumnsToContents()
            header_h = table.horizontalHeader().height()
            rows_h   = sum(table.rowHeight(i) for i in range(n_shots))
            table.setFixedHeight(header_h + rows_h + 4)

            gb_layout.addWidget(table)
            cont_layout.addWidget(gb)

        cont_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                   Qt.Horizontal, self)
        buttons.button(QDialogButtonBox.Ok).setText("Confirm Selections")
        buttons.accepted.connect(self._confirm)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _confirm(self):
        for group, btn_group in self._btn_groups:
            if btn_group.checkedId() == -1:
                QMessageBox.warning(
                    self, "Incomplete Selection",
                    f"Please select a shot to keep for "
                    f"Line {group['line']}, Station {group['station']}."
                )
                return
        self.accept()

    def get_selections(self):
        return {
            (group['line'], group['station']): group['shots'][btn_group.checkedId()]['file_num']
            for group, btn_group in self._btn_groups
            if btn_group.checkedId() >= 0
        }


# ── Stn Num Check tab ─────────────────────────────────────────────────────────

class VizTab(QWidget):
    """Station table (editable) + sequence chart + map."""

    log = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._date_part      = None
        self._obs_df         = pd.DataFrame()
        self._cog_df         = pd.DataFrame()
        self._orig_stations  = {}   # file_num (str) → original station str
        self._sps_df         = None
        self._settings       = QSettings("SeismicQC", "LogChecker")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        # ── SPS file bar ──────────────────────────────────────────────────────
        sps_row = QHBoxLayout()
        sps_row.addWidget(QLabel("SPS File:"))
        self._sps_label = QLabel("No file loaded")
        self._sps_label.setStyleSheet("color: grey;")
        sps_row.addWidget(self._sps_label, 1)
        self._sps_btn = QPushButton("Load SPS…")
        self._sps_btn.setFixedWidth(100)
        self._sps_btn.clicked.connect(self._load_sps)
        sps_row.addWidget(self._sps_btn)
        outer.addLayout(sps_row)

        # Shown when no data yet
        self._placeholder = QLabel("Run QC to populate charts.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._placeholder)

        # Restore last SPS path
        last_sps = self._settings.value("last_sps_path", "")
        if last_sps and Path(last_sps).exists():
            self._load_sps_from_path(last_sps)

        # Main splitter: left = table panel, right = charts panel
        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.hide()
        outer.addWidget(self._splitter)

        # ── Left panel ────────────────────────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["File#", "Line", "Station"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setFont(QFont("Courier New", 9))
        left_layout.addWidget(self._table)

        self._apply_btn = QPushButton("Apply Changes")
        self._apply_btn.clicked.connect(self._apply_corrections)
        left_layout.addWidget(self._apply_btn)

        # ── Right panel ───────────────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.fig    = Figure(tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self._toolbar = NavToolbar(self.canvas, self)
        right_layout.addWidget(self._toolbar)
        right_layout.addWidget(self.canvas)

        self._splitter.addWidget(left)
        self._splitter.addWidget(right)
        self._splitter.setSizes([260, 740])

        # Hover annotation state
        self._annot_map    = None
        self._scatter_data = []
        self.canvas.mpl_connect('motion_notify_event', self._on_hover)

    # ── Public ────────────────────────────────────────────────────────────────

    def update_plots(self, obs_df: pd.DataFrame, cog_df: pd.DataFrame, date_part: str):
        self._date_part = date_part
        self._obs_df    = obs_df.copy()
        self._cog_df    = cog_df.copy()
        self._populate_table()
        self._render_plots()
        self._placeholder.hide()
        self._splitter.show()

    # ── Table ─────────────────────────────────────────────────────────────────

    def _populate_table(self):
        self._orig_stations = {}
        obs = self._obs_df
        self._table.setRowCount(len(obs))
        for row_i, (_, r) in enumerate(obs.iterrows()):
            fn  = str(int(float(r['File#'])))
            ln  = str(r['Line'])
            stn = str(r['Station'])

            fn_item = QTableWidgetItem(fn)
            fn_item.setFlags(fn_item.flags() & ~Qt.ItemIsEditable)
            ln_item = QTableWidgetItem(ln)
            ln_item.setFlags(ln_item.flags() & ~Qt.ItemIsEditable)
            stn_item = QTableWidgetItem(stn)

            self._table.setItem(row_i, 0, fn_item)
            self._table.setItem(row_i, 1, ln_item)
            self._table.setItem(row_i, 2, stn_item)

            self._orig_stations[fn] = stn

        self._table.resizeColumnsToContents()

    def _load_sps(self):
        start = str(Path(self._settings.value("last_sps_path", "")) .parent) \
                if self._settings.value("last_sps_path", "") else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SPS File", start, "SPS Files (*.sps);;All Files (*)"
        )
        if path:
            self._load_sps_from_path(path)

    def _load_sps_from_path(self, path):
        try:
            from file_io import read_sps
            self._sps_df = read_sps(path)
            n = len(self._sps_df)
            self._sps_label.setText(f"{Path(path).name}  ({n:,} points)")
            self._sps_label.setStyleSheet("color: green;")
            self._settings.setValue("last_sps_path", path)
            self.log.emit(f"  SPS loaded: {Path(path).name} ({n:,} points)")
        except Exception as exc:
            self._sps_df = None
            self._sps_label.setText(f"Failed to load: {Path(path).name}")
            self._sps_label.setStyleSheet("color: red;")
            QMessageBox.critical(self, "SPS Load Error", str(exc))

    def _apply_corrections(self):
        import traceback

        # Commit any active cell editor before reading values
        self._table.setCurrentItem(None)

        corrections = {}
        for row_i in range(self._table.rowCount()):
            fn      = self._table.item(row_i, 0).text()
            new_stn = self._table.item(row_i, 2).text().strip()
            if new_stn == self._orig_stations.get(fn):
                continue
            try:
                corrections[fn] = float(new_stn)
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid Value",
                    f"Station '{new_stn}' for File# {fn} is not numeric.",
                )
                return

        if not corrections:
            QMessageBox.information(self, "No Changes", "No station numbers were changed.")
            return

        # Log header
        self.log.emit("\n══════════════════════════════════════════")
        self.log.emit("  Stn Num Check — Applying Corrections")
        self.log.emit("══════════════════════════════════════════")
        for fn, new_stn in corrections.items():
            old_stn = self._orig_stations.get(fn, '?')
            self.log.emit(f"  File# {fn}: Station {old_stn} → {new_stn}")

        # Update in-memory obs_df, refresh tracking, and re-render — always
        for fn, new_stn in corrections.items():
            self._obs_df.loc[self._obs_df['File#'].astype(str) == fn, 'Station'] = new_stn
        for fn, new_stn in corrections.items():
            self._orig_stations[fn] = str(new_stn)
        self._render_plots()

        # Write to disk
        try:
            from file_io import apply_station_corrections
            apply_station_corrections(corrections, self._date_part, QC_DIR, REMOVED_DIR,
                                      log_fn=self.log.emit, sps_df=self._sps_df)
        except Exception:
            msg = traceback.format_exc()
            self.log.emit(f"\n  ERROR — disk write failed:\n{msg}")
            QMessageBox.critical(
                self, "Write Error",
                f"Plots updated in memory but disk write failed:\n\n{msg}",
            )
            return

        self.log.emit("══════════════════════════════════════════\n")
        n = len(corrections)
        QMessageBox.information(
            self, "Applied",
            f"{n} correction{'s' if n > 1 else ''} saved and plots updated.",
        )

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_plots(self):
        self.fig.clear()
        self._scatter_data = []
        self._annot_map    = None

        obs_df = self._obs_df
        cog_df = self._cog_df

        if obs_df.empty:
            self.canvas.draw()
            return

        # Merge OBS and COG on File# == FF ID
        merged = obs_df.copy()
        merged['_key'] = merged['File#'].astype(str)

        if not cog_df.empty:
            cog = cog_df.copy()
            cog['_key'] = cog['FF ID'].astype(str)
            merged = merged.merge(
                cog[['_key', 'Decoder Lat', 'Decoder Lon', 'Distance to Source Point']],
                on='_key', how='left',
            )
        else:
            merged['Decoder Lat'] = pd.NA
            merged['Decoder Lon'] = pd.NA
            merged['Distance to Source Point'] = pd.NA

        lines  = sorted(merged['Line'].unique())
        colors = [cm.tab10(i / max(len(lines), 1)) for i in range(len(lines))]

        ax_seq = self.fig.add_subplot(1, 2, 1)
        ax_map = self.fig.add_subplot(1, 2, 2)

        for color, line in zip(colors, lines):
            grp = merged[merged['Line'] == line].sort_values('File#')

            ax_seq.plot(
                grp['File#'], grp['Station'],
                'o-', markersize=3, linewidth=0.7,
                color=color, label=f'Line {line}',
            )

            has_coords = grp['Decoder Lat'].notna() & grp['Decoder Lon'].notna()
            grp_geo    = grp[has_coords]
            if not grp_geo.empty:
                sc = ax_map.scatter(
                    grp_geo['Decoder Lon'], grp_geo['Decoder Lat'],
                    s=18, color=color, label=f'Line {line}',
                    zorder=3, picker=False,
                )
                labels = [
                    f"File#: {r['File#']}\nLine: {r['Line']}\n"
                    f"Station: {r['Station']}\n"
                    f"Dist to SP: {r.get('Distance to Source Point', 'N/A')}"
                    for _, r in grp_geo.iterrows()
                ]
                lons = grp_geo['Decoder Lon'].to_numpy()
                lats = grp_geo['Decoder Lat'].to_numpy()
                self._scatter_data.append((sc, lons, lats, labels))

        ax_seq.set_xlabel('File#')
        ax_seq.set_ylabel('Station')
        ax_seq.set_title('Station Sequence')
        ax_seq.grid(True, alpha=0.3)
        if len(lines) > 1:
            ax_seq.legend(fontsize=7)

        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')
        ax_map.set_title('Shot Positions')
        ax_map.grid(True, alpha=0.3)
        if len(lines) > 1:
            ax_map.legend(fontsize=7)

        self._annot_map = ax_map.annotate(
            "", xy=(0, 0), xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9,
                      ec="grey", lw=0.8),
            fontsize=8,
        )
        self._annot_map.set_visible(False)

        self.canvas.draw()

    # ── Hover ─────────────────────────────────────────────────────────────────

    def _on_hover(self, event):
        if self._annot_map is None or event.inaxes is not self._annot_map.axes:
            return

        import numpy as np
        found = False
        for sc, lons, lats, labels in self._scatter_data:
            if not len(lons):
                continue
            ax   = sc.axes
            xy   = ax.transData.transform(list(zip(lons, lats)))
            ex, ey = event.x, event.y
            dist = ((xy[:, 0] - ex) ** 2 + (xy[:, 1] - ey) ** 2) ** 0.5
            idx  = int(dist.argmin())
            if dist[idx] < 12:
                self._annot_map.set_text(labels[idx])
                self._annot_map.xy = (lons[idx], lats[idx])
                self._annot_map.set_visible(True)
                self.canvas.draw_idle()
                found = True
                break

        if not found and self._annot_map.get_visible():
            self._annot_map.set_visible(False)
            self.canvas.draw_idle()


# ── Vibe QC tab ───────────────────────────────────────────────────────────────

# Section config: list of (col_names, line_labels, linestyles, alphas, ylabel)
_PERF_SECTIONS = [
    (['Phase Max', 'Phase Avg'], ['Max', 'Avg'], ['-', '--'], [0.9, 0.6], 'Phase (°)'),
    (['Force Max', 'Force Avg'], ['Max', 'Avg'], ['-', '--'], [0.9, 0.6], 'Force (%)'),
    (['THD Max',   'THD Avg'],   ['Max', 'Avg'], ['-', '--'], [0.9, 0.6], 'THD (%)'),
]

_GPS_SECTIONS = [
    (['Sats'],         ['Sats'],         ['-'],       [0.9],        'Satellites'),
    (['PDOP', 'HDOP'], ['PDOP', 'HDOP'], ['-', '--'], [0.9, 0.7],   'DOP'),
]

_GROUND_SECTIONS = [
    (['Max Viscosity', 'Min Viscosity', 'Avg Viscosity'],
     ['Max', 'Min', 'Avg'], ['-', ':', '--'], [0.9, 0.6, 0.7], 'Viscosity'),
    (['Max Stiffness', 'Min Stiffness', 'Avg Stiffness'],
     ['Max', 'Min', 'Avg'], ['-', ':', '--'], [0.9, 0.6, 0.7], 'Stiffness'),
    (['Drive Level'], ['Drive Level'], ['-'], [0.9], 'Drive Level (%)'),
]

_NUMERIC_COLS = [
    'File Num', 'Unit ID',
    'Phase Max', 'Phase Avg', 'Force Max', 'Force Avg', 'THD Max', 'THD Avg',
    'Sats', 'PDOP', 'HDOP', 'VDOP',
    'Max Viscosity', 'Min Viscosity', 'Avg Viscosity',
    'Max Stiffness', 'Min Stiffness', 'Avg Stiffness',
    'Drive Level',
]


class VibeTab(QWidget):
    """Per-vibe performance, GPS quality, and ground coupling charts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.transforms import blended_transform_factory
        self._blended = blended_transform_factory  # stored for use in _render_grid

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Top bar ───────────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.addWidget(QLabel("Dataset:"))
        self._day_combo = QComboBox()
        self._day_combo.setMinimumWidth(240)
        top.addWidget(self._day_combo)
        load_btn = QPushButton("Load")
        load_btn.setFixedWidth(60)
        load_btn.clicked.connect(self._load)
        top.addWidget(load_btn)

        self._pdf_btn = QPushButton("Export PDF Report")
        self._pdf_btn.setFixedWidth(150)
        self._pdf_btn.setEnabled(False)
        self._pdf_btn.clicked.connect(self._export_pdf)
        top.addWidget(self._pdf_btn)

        top.addStretch()
        layout.addLayout(top)

        self._populate_combo()

        # ── Inner tabs ────────────────────────────────────────────────────────
        self._inner = QTabWidget()
        layout.addWidget(self._inner)

        (self._perf_widget,   self._perf_fig,   self._perf_canvas)   = self._make_chart_widget()
        (self._gps_widget,    self._gps_fig,    self._gps_canvas)    = self._make_chart_widget()
        (self._ground_widget, self._ground_fig, self._ground_canvas) = self._make_chart_widget()
        (self._pos_widget,    self._pos_fig,    self._pos_canvas)    = self._make_chart_widget()

        self._inner.addTab(self._perf_widget,   "Performance (Phase / Force / THD)")
        self._inner.addTab(self._gps_widget,    "GPS Quality (Sats / DOP)")
        self._inner.addTab(self._ground_widget, "Ground Coupling (Viscosity / Stiffness / Drive)")
        self._inner.addTab(self._pos_widget,    "Vibe Positioning")

        tb = self._inner.tabBar()
        tb.setTabToolTip(0,
            "Phase  — how closely the output sweep matches the reference signal (°); lower = better\n"
            "Force  — hydraulic output force as a % of the target force\n"
            "THD    — Total Harmonic Distortion: unwanted frequency content in the sweep (%); lower = better"
        )
        tb.setTabToolTip(1,
            "Sats   — number of GPS satellites tracked; more = better positioning\n"
            "PDOP   — Position Dilution of Precision: overall GPS geometry quality; lower = better\n"
            "HDOP   — Horizontal Dilution of Precision: horizontal GPS accuracy; lower = better"
        )
        tb.setTabToolTip(2,
            "Viscosity  — fluid damping at the vibe baseplate / ground interface; indicates ground softness\n"
            "Stiffness  — ground rigidity at the vibe baseplate interface; higher = harder ground\n"
            "Drive Level — hydraulic force the vibrator is applying (% of max); flat line is normal"
        )
        tb.setTabToolTip(3,
            "Offset      — distance each shot fired from its planned source point (m)\n"
            "Separation  — distance between vibes on the same shot (multi-vibe days only)\n"
            "Map         — actual vibe positions vs planned source points"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_chart_widget(self):
        widget  = QWidget()
        vbox    = QVBoxLayout(widget)
        vbox.setContentsMargins(0, 0, 0, 0)
        fig     = Figure()
        canvas  = FigureCanvas(fig)
        toolbar = NavToolbar(canvas, widget)
        vbox.addWidget(toolbar)
        vbox.addWidget(canvas)
        return widget, fig, canvas

    # ── Load / Refresh ────────────────────────────────────────────────────────

    def _populate_combo(self, select_date_part=None):
        """Rebuild the dropdown from QC_DIR. Optionally select a specific date_part."""
        self._day_combo.blockSignals(True)
        self._day_combo.clear()

        daily_dirs = sorted(
            (d for d in QC_DIR.iterdir()
             if d.is_dir() and d.name != 'combined'
             and (d / f'PSS_QC_{d.name}.csv').exists()),
            reverse=True,
        )
        for d in daily_dirs:
            parts = d.name.split('_')           # ['2026','04','14','06','50','12']
            if len(parts) >= 3:
                label = f"{parts[0]}-{parts[1]}-{parts[2]}  {parts[3]}:{parts[4]}" \
                        if len(parts) >= 5 else d.name
            else:
                label = d.name
            self._day_combo.addItem(label, userData=d.name)

        combined_path = QC_DIR / 'combined' / 'PSS_QC_Combined.csv'
        if combined_path.exists():
            self._day_combo.addItem('── Combined (all days) ──', userData='combined')

        # Restore selection
        if select_date_part:
            for i in range(self._day_combo.count()):
                if self._day_combo.itemData(i) == select_date_part:
                    self._day_combo.setCurrentIndex(i)
                    break

        self._day_combo.blockSignals(False)

    def refresh(self, pss_df, cog_df=None, date_part=None):
        """Called after a QC run — update dropdown and render the new day's data."""
        self._populate_combo(select_date_part=date_part)
        self._render(pss_df.copy(), cog_df.copy() if cog_df is not None else pd.DataFrame())

    def _load(self):
        key = self._day_combo.currentData()
        if not key:
            return
        if key == 'combined':
            pss_path = QC_DIR / 'combined' / 'PSS_QC_Combined.csv'
            cog_path = QC_DIR / 'combined' / 'FinalCOG_QC_Combined.csv'
        else:
            pss_path = QC_DIR / key / f'PSS_QC_{key}.csv'
            cog_path = QC_DIR / key / f'FinalCOG_QC_{key}.csv'
        if not pss_path.exists():
            QMessageBox.warning(self, "File Not Found", f"PSS file not found:\n{pss_path}")
            return
        try:
            pss_df = pd.read_csv(pss_path)
            cog_df = pd.read_csv(cog_path) if cog_path.exists() else pd.DataFrame()
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return
        self._render(pss_df, cog_df)

    def _render(self, df, cog_df=None):
        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        unit_ids = sorted(df['Unit ID'].dropna().unique())
        colors   = [cm.tab10(i / max(len(unit_ids), 1)) for i in range(len(unit_ids))]

        self._render_grid(self._perf_fig,   self._perf_canvas,
                          df, unit_ids, colors, _PERF_SECTIONS,
                          'Vibrator Performance  —  Phase / Force / THD')
        self._render_gps(df, unit_ids, colors)
        self._render_grid(self._ground_fig, self._ground_canvas,
                          df, unit_ids, colors, _GROUND_SECTIONS,
                          'Ground Coupling  —  Viscosity / Stiffness / Drive Level')
        self._render_positioning(df, cog_df if cog_df is not None else pd.DataFrame(),
                                 unit_ids, colors)
        self._pdf_btn.setEnabled(True)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _export_pdf(self):
        from datetime import datetime
        from matplotlib.backends.backend_pdf import PdfPages

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        default_name = f"VibeQC_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", default_name, "PDF Files (*.pdf)"
        )
        if not path:
            return

        A4L = (11.69, 8.27)   # A4 landscape inches

        try:
            with PdfPages(path) as pdf:
                # ── Cover page ────────────────────────────────────────────────
                fig_cover = Figure(figsize=A4L)
                ax = fig_cover.add_subplot(111)
                ax.axis('off')
                ax.text(0.5, 0.62, 'Vibe QC Report',
                        ha='center', va='center',
                        fontsize=28, fontweight='bold', transform=ax.transAxes)
                ax.text(0.5, 0.50, 'Vibrator Performance Analysis',
                        ha='center', va='center',
                        fontsize=14, color='#444444', transform=ax.transAxes)
                ax.text(0.5, 0.38, f'Generated: {timestamp}',
                        ha='center', va='center',
                        fontsize=11, color='#666666', transform=ax.transAxes)
                pdf.savefig(fig_cover, bbox_inches='tight')
                plt.close(fig_cover)

                # ── Chart pages ───────────────────────────────────────────────
                pages = [
                    (self._perf_fig,   'Performance  —  Phase / Force / THD'),
                    (self._gps_fig,    'GPS Quality  —  Satellites / DOP / Fix Type'),
                    (self._ground_fig, 'Ground Coupling  —  Viscosity / Stiffness / Drive Level'),
                ]
                for fig, page_title in pages:
                    orig_size = fig.get_size_inches()
                    fig.set_size_inches(A4L)
                    fig.tight_layout(rect=[0, 0, 0.97, 0.96])
                    pdf.savefig(fig, bbox_inches='tight')
                    fig.set_size_inches(orig_size)
                    fig.tight_layout(rect=[0, 0, 0.97, 0.96])

                # ── PDF metadata ──────────────────────────────────────────────
                d = pdf.infodict()
                d['Title']        = 'Vibe QC Report'
                d['Author']       = 'Seismic Log QC Checker'
                d['Subject']      = 'Vibrator Performance Analysis'
                d['CreationDate'] = datetime.now()

            # Redraw canvases to restore on-screen appearance
            for canvas in (self._perf_canvas, self._gps_canvas, self._ground_canvas):
                canvas.draw()

            QMessageBox.information(self, "PDF Saved", f"Report saved to:\n{path}")

        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Export Error", traceback.format_exc())

    def _render_positioning(self, pss_df, cog_df, unit_ids, colors):
        import numpy as np
        import math

        fig    = self._pos_fig
        canvas = self._pos_canvas
        fig.clear()

        threshold = 5.0   # metres

        # ── Row 0: offset from planned source point (from COG) ────────────────
        ax_off = fig.add_subplot(3, 1, 1)
        has_offset = (not cog_df.empty
                      and 'FF ID' in cog_df.columns
                      and 'Distance to Source Point' in cog_df.columns)
        if has_offset:
            cog = cog_df.copy()
            cog['FF ID'] = pd.to_numeric(cog['FF ID'], errors='coerce')
            cog['Distance to Source Point'] = pd.to_numeric(
                cog['Distance to Source Point'], errors='coerce')
            cog = cog.dropna(subset=['FF ID', 'Distance to Source Point'])
            cog = cog[cog['Distance to Source Point'] < 1e6].sort_values('FF ID')

            ok  = cog[cog['Distance to Source Point'] <= threshold]
            bad = cog[cog['Distance to Source Point'] >  threshold]
            ax_off.scatter(ok['FF ID'],  ok['Distance to Source Point'],
                           s=8, color='steelblue', label='Within threshold', zorder=3)
            ax_off.scatter(bad['FF ID'], bad['Distance to Source Point'],
                           s=12, color='red', label='Exceeds threshold', zorder=4)
            ax_off.axhline(threshold, color='orange', linewidth=0.9,
                           linestyle='--', label=f'Threshold ({threshold:.0f} m)')
            ax_off.set_ylabel('Offset (m)', fontsize=8)
            ax_off.legend(fontsize=6, loc='upper right')
        else:
            ax_off.text(0.5, 0.5, 'No COG offset data available',
                        ha='center', va='center', transform=ax_off.transAxes, fontsize=9)
        ax_off.set_title('Offset from Planned Source Point', fontsize=9)
        ax_off.grid(True, alpha=0.25)
        ax_off.tick_params(labelsize=7)
        plt.setp(ax_off.get_xticklabels(), visible=False)

        # ── Row 1: vibe-to-vibe separation ────────────────────────────────────
        ax_sep = fig.add_subplot(3, 1, 2)
        pss_df['Lat'] = pd.to_numeric(pss_df['Lat'], errors='coerce')
        pss_df['Lon'] = pd.to_numeric(pss_df['Lon'], errors='coerce')
        pss_df['File Num'] = pd.to_numeric(pss_df['File Num'], errors='coerce')
        multi = pss_df.dropna(subset=['File Num', 'Lat', 'Lon'])
        grp_sizes = multi.groupby('File Num').size()
        multi_file_nums = grp_sizes[grp_sizes >= 2].index

        if len(multi_file_nums):
            seps = []
            for fn, grp in multi.groupby('File Num'):
                if fn not in multi_file_nums:
                    continue
                lats = grp['Lat'].values
                lons = grp['Lon'].values
                # Distance between first two vibes
                R    = 6_371_000.0
                dlat = math.radians(lats[1] - lats[0])
                dlon = math.radians(lons[1] - lons[0])
                a    = (math.sin(dlat / 2) ** 2
                        + math.cos(math.radians(lats[0]))
                        * math.cos(math.radians(lats[1]))
                        * math.sin(dlon / 2) ** 2)
                seps.append({'File Num': fn,
                             'Separation': R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))})
            sep_df = pd.DataFrame(seps).sort_values('File Num')
            ax_sep.plot(sep_df['File Num'], sep_df['Separation'],
                        'o-', markersize=3, linewidth=0.6, color='steelblue')
            mean_sep = sep_df['Separation'].mean()
            ax_sep.axhline(mean_sep, color='grey', linewidth=0.8, linestyle=':')
            ax_sep.text(sep_df['File Num'].iloc[-1], mean_sep,
                        f' mean {mean_sep:.1f}m', va='bottom', fontsize=6, color='grey')
            ax_sep.set_ylabel('Separation (m)', fontsize=8)
        else:
            ax_sep.text(0.5, 0.5, 'Single-vibe day — no separation data',
                        ha='center', va='center', transform=ax_sep.transAxes, fontsize=9)
        ax_sep.set_title('Vibe-to-Vibe Separation', fontsize=9)
        ax_sep.grid(True, alpha=0.25)
        ax_sep.tick_params(labelsize=7)
        plt.setp(ax_sep.get_xticklabels(), visible=False)

        # ── Row 2: map — actual positions vs planned ───────────────────────────
        ax_map = fig.add_subplot(3, 1, 3)
        for color, uid in zip(colors, unit_ids):
            grp = pss_df[pss_df['Unit ID'] == uid].dropna(subset=['Lat', 'Lon'])
            if not grp.empty:
                ax_map.scatter(grp['Lon'], grp['Lat'], s=6, color=color,
                               label=f'Unit {int(uid)}', zorder=3, alpha=0.7)

        if has_offset and 'Source Point Lat' in cog_df.columns and 'Source Point Lon' in cog_df.columns:
            sp_lat = pd.to_numeric(cog_df['Source Point Lat'], errors='coerce').dropna()
            sp_lon = pd.to_numeric(cog_df['Source Point Lon'], errors='coerce').dropna()
            valid  = sp_lat.index.intersection(sp_lon.index)
            ax_map.scatter(sp_lon[valid], sp_lat[valid], s=4, color='lightgrey',
                           marker='+', label='Planned', zorder=2)

        ax_map.set_xlabel('Longitude', fontsize=7)
        ax_map.set_ylabel('Latitude',  fontsize=7)
        ax_map.set_title('Actual Vibe Positions vs Planned', fontsize=9)
        ax_map.grid(True, alpha=0.25)
        ax_map.tick_params(labelsize=7)
        ax_map.legend(fontsize=6, loc='upper right')

        fig.suptitle('Vibe Positioning', fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        canvas.draw()

    def _render_gps(self, df, unit_ids, colors):
        from matplotlib.transforms import blended_transform_factory
        fig    = self._gps_fig
        canvas = self._gps_canvas
        fig.clear()

        n_cols = len(unit_ids)

        for col_i, (uid, color) in enumerate(zip(unit_ids, colors)):
            grp = df[df['Unit ID'] == uid].sort_values('File Num')

            # ── Row 0: Sats ───────────────────────────────────────────────────
            ax = fig.add_subplot(3, n_cols, col_i + 1)
            t  = blended_transform_factory(ax.transAxes, ax.transData)
            s  = pd.to_numeric(grp['Sats'], errors='coerce')
            ax.plot(grp['File Num'], s, '-', color=color, linewidth=0.9)
            mean_val = s.mean()
            ax.axhline(mean_val, color=color, linewidth=0.8, linestyle=':', alpha=0.8)
            ax.text(1.01, mean_val, f'{mean_val:.1f}', transform=t,
                    va='center', ha='left', fontsize=6, color=color, clip_on=False)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.set_title(f'Unit {int(uid)}', fontsize=9, color=color)
            if col_i == 0:
                ax.set_ylabel('Satellites', fontsize=8)

            # ── Row 1: PDOP + HDOP ────────────────────────────────────────────
            ax = fig.add_subplot(3, n_cols, n_cols + col_i + 1)
            t  = blended_transform_factory(ax.transAxes, ax.transData)
            for col_name, style, alpha, lbl in [
                ('PDOP', '-',  0.9, 'PDOP'),
                ('HDOP', '--', 0.7, 'HDOP'),
            ]:
                s = pd.to_numeric(grp[col_name], errors='coerce')
                ax.plot(grp['File Num'], s, style, color=color,
                        linewidth=0.9, alpha=alpha, label=lbl)
                mean_val = s.mean()
                ax.axhline(mean_val, color=color, linewidth=0.8,
                           linestyle=':', alpha=alpha * 0.8)
                ax.text(1.01, mean_val, f'{mean_val:.1f}', transform=t,
                        va='center', ha='left', fontsize=6,
                        color=color, alpha=alpha, clip_on=False)
            ax.legend(fontsize=6, loc='upper right',
                      handlelength=1.5, borderpad=0.4)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)
            plt.setp(ax.get_xticklabels(), visible=False)
            if col_i == 0:
                ax.set_ylabel('DOP', fontsize=8)

            # ── Row 2: Quality tally (horizontal bar chart) ───────────────────
            ax = fig.add_subplot(3, n_cols, 2 * n_cols + col_i + 1)
            qual_counts = grp['Quality'].value_counts()
            bars = ax.barh(qual_counts.index.astype(str), qual_counts.values,
                           color=color, alpha=0.8)
            ax.bar_label(bars, fontsize=6, padding=3)
            ax.set_xlim(right=qual_counts.values.max() * 1.2)
            ax.tick_params(labelsize=7)
            ax.set_xlabel('Count', fontsize=7)
            ax.grid(True, alpha=0.25, axis='x')
            if col_i == 0:
                ax.set_ylabel('GPS Fix Type', fontsize=8)

        fig.suptitle('GPS Quality  —  Satellites / DOP / Fix Type', fontsize=9)
        fig.tight_layout(rect=[0, 0, 0.97, 0.97])
        canvas.draw()

    def _render_grid(self, fig, canvas, df, unit_ids, colors, sections, title):
        from matplotlib.transforms import blended_transform_factory
        fig.clear()
        n_rows   = len(sections)
        n_cols   = len(unit_ids)

        for col_i, (uid, color) in enumerate(zip(unit_ids, colors)):
            grp = df[df['Unit ID'] == uid].sort_values('File Num')

            for row_i, (col_names, line_labels, styles, alphas, ylabel) in enumerate(sections):
                ax = fig.add_subplot(n_rows, n_cols, row_i * n_cols + col_i + 1)
                t  = blended_transform_factory(ax.transAxes, ax.transData)

                for col_name, label, style, alpha in zip(col_names, line_labels, styles, alphas):
                    if col_name not in grp.columns:
                        continue
                    series = pd.to_numeric(grp[col_name], errors='coerce')
                    if series.isna().all():
                        continue
                    ax.plot(grp['File Num'], series,
                            style, color=color, linewidth=0.9,
                            alpha=alpha, label=label)
                    mean_val = series.mean()
                    ax.axhline(mean_val, color=color, linewidth=0.8,
                               linestyle=':', alpha=alpha * 0.85)
                    ax.text(1.01, mean_val, f'{mean_val:.1f}',
                            transform=t, va='center', ha='left',
                            fontsize=6, color=color, alpha=alpha,
                            clip_on=False)

                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=7)
                if len(col_names) > 1:
                    ax.legend(fontsize=6, loc='upper right',
                              handlelength=1.5, borderpad=0.4)

                if row_i == 0:
                    ax.set_title(f'Unit {int(uid)}', fontsize=9, color=color)
                if col_i == 0:
                    ax.set_ylabel(ylabel, fontsize=8)
                if row_i == n_rows - 1:
                    ax.set_xlabel('File #', fontsize=7)
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)

        fig.suptitle(title, fontsize=9)
        fig.tight_layout(rect=[0, 0, 0.97, 0.97])   # right margin for mean labels
        canvas.draw()


# ── Combined File Check tab ───────────────────────────────────────────────────

class CombinedCheckTab(QWidget):
    """Scan combined OBS for (Line, Station) duplicates and remove them."""

    log = pyqtSignal(str)

    _COLS = [
        ("File#",          "File#"),
        ("Line",           "Line"),
        ("Station",        "Station"),
        ("Local Date",     " Local Date"),
        ("Local Time",     " Local Time"),
        ("PSS Info",       " PSS Info"),
        ("Phase Max",      "_Phase Max"),
        ("Force Max",      "_Force Max"),
        ("THD Max",        "_THD Max"),
        ("Dist to SP (m)", "_Dist to SP"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._btn_groups = []   # list of (shots_list, QButtonGroup)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Top bar ───────────────────────────────────────────────────────────
        top = QHBoxLayout()
        check_btn = QPushButton("Check for Duplicates")
        check_btn.setFixedWidth(180)
        check_btn.clicked.connect(self._check)
        top.addWidget(check_btn)
        self._status_lbl = QLabel("")
        top.addWidget(self._status_lbl)
        top.addStretch()
        layout.addLayout(top)

        # ── Scroll area ───────────────────────────────────────────────────────
        self._scroll    = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._container = QWidget()
        self._cont_lay  = QVBoxLayout(self._container)
        self._cont_lay.setSpacing(10)
        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll)

        # ── Apply button ──────────────────────────────────────────────────────
        self._apply_btn = QPushButton("Apply Removals")
        self._apply_btn.setFixedWidth(150)
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self._apply)
        layout.addWidget(self._apply_btn, alignment=Qt.AlignLeft)

    # ── Check ─────────────────────────────────────────────────────────────────

    def _check(self):
        combined_obs = QC_DIR / 'combined' / 'ObserverLog_Detailed_QC_Combined.csv'
        if not combined_obs.exists():
            QMessageBox.warning(self, "Not Found",
                                "Combined OBS file not found.\nRun QC first.")
            return

        obs = pd.read_csv(combined_obs, skiprows=2)

        # Join PSS metrics (max per File#) from combined PSS
        pss_path = QC_DIR / 'combined' / 'PSS_QC_Combined.csv'
        if pss_path.exists():
            pss = pd.read_csv(pss_path)
            for col in ['Phase Max', 'Force Max', 'THD Max']:
                pss[col] = pd.to_numeric(pss[col], errors='coerce')
            pss_agg = (pss.groupby('File Num')[['Phase Max', 'Force Max', 'THD Max']]
                          .max().reset_index()
                          .rename(columns={'File Num': 'File#',
                                           'Phase Max': '_Phase Max',
                                           'Force Max': '_Force Max',
                                           'THD Max':   '_THD Max'}))
            pss_agg['File#'] = pd.to_numeric(pss_agg['File#'], errors='coerce')
            obs['File#']     = pd.to_numeric(obs['File#'], errors='coerce')
            obs = obs.merge(pss_agg, on='File#', how='left')

        # Join COG Distance to Source Point
        cog_path = QC_DIR / 'combined' / 'FinalCOG_QC_Combined.csv'
        if cog_path.exists():
            cog = pd.read_csv(cog_path)[['FF ID', 'Distance to Source Point']]
            cog['Distance to Source Point'] = pd.to_numeric(
                cog['Distance to Source Point'], errors='coerce')
            # Filter out bad decoder rows (distance > 1e6 means decoder had no GPS fix)
            cog.loc[cog['Distance to Source Point'] > 1e6,
                    'Distance to Source Point'] = pd.NA
            cog = cog.rename(columns={'FF ID': 'File#',
                                      'Distance to Source Point': '_Dist to SP'})
            cog['File#'] = pd.to_numeric(cog['File#'], errors='coerce')
            obs = obs.merge(cog, on='File#', how='left')

        dup_mask = obs.duplicated(subset=['Line', 'Station'], keep=False)
        dups     = obs[dup_mask]

        # Clear previous results
        while self._cont_lay.count():
            item = self._cont_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._btn_groups = []
        self._apply_btn.setEnabled(False)

        if dups.empty:
            self._status_lbl.setText("No duplicates found.")
            self._status_lbl.setStyleSheet("color: green;")
            lbl = QLabel("No duplicates found in combined files.")
            lbl.setAlignment(Qt.AlignCenter)
            self._cont_lay.addWidget(lbl)
            self._cont_lay.addStretch()
            return

        n_groups = dups.groupby(['Line', 'Station']).ngroups
        self._status_lbl.setText(
            f"{n_groups} duplicate group{'s' if n_groups > 1 else ''} found.")
        self._status_lbl.setStyleSheet("color: red;")

        for (line, station), grp in dups.groupby(['Line', 'Station']):
            gb        = QGroupBox(f"Line: {line}   \u00b7   Station: {station}")
            gb_layout = QVBoxLayout(gb)
            n_shots   = len(grp)
            n_cols    = len(self._COLS) + 1

            table = QTableWidget(n_shots, n_cols)
            table.setHorizontalHeaderLabels(["Keep"] + [c[0] for c in self._COLS])
            table.verticalHeader().setVisible(False)
            table.setSelectionMode(QTableWidget.NoSelection)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.horizontalHeader().setStretchLastSection(False)

            btn_group  = QButtonGroup(self)
            shots_list = []
            btn_group.buttonClicked.connect(self._on_selection_changed)

            for row_i, (_, row) in enumerate(grp.iterrows()):
                fn = str(row['File#'])
                shots_list.append(fn)

                radio_wrap = QWidget()
                radio_hbox = QHBoxLayout(radio_wrap)
                radio_hbox.setContentsMargins(4, 0, 4, 0)
                radio_hbox.setAlignment(Qt.AlignCenter)
                radio = QRadioButton()
                btn_group.addButton(radio, row_i)
                radio_hbox.addWidget(radio)
                table.setCellWidget(row_i, 0, radio_wrap)

                for col_i, (_, key) in enumerate(self._COLS):
                    val  = str(row.get(key, 'N/A')).strip()
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignCenter)
                    table.setItem(row_i, col_i + 1, item)

            table.resizeColumnsToContents()
            header_h = table.horizontalHeader().height()
            rows_h   = sum(table.rowHeight(i) for i in range(n_shots))
            table.setFixedHeight(header_h + rows_h + 4)

            self._btn_groups.append((shots_list, btn_group))
            gb_layout.addWidget(table)
            self._cont_lay.addWidget(gb)

        self._cont_lay.addStretch()

    # ── Selection changed ─────────────────────────────────────────────────────

    def _on_selection_changed(self):
        all_resolved = bool(self._btn_groups) and all(
            bg.checkedId() >= 0 for _, bg in self._btn_groups
        )
        self._apply_btn.setEnabled(all_resolved)

    # ── Apply ─────────────────────────────────────────────────────────────────

    def _apply(self):
        discard_file_nums = set()
        for shots_list, btn_group in self._btn_groups:
            keep_idx = btn_group.checkedId()
            for i, fn in enumerate(shots_list):
                if i != keep_idx:
                    discard_file_nums.add(fn)

        if not discard_file_nums:
            return

        self.log.emit("\n══════════════════════════════════════════")
        self.log.emit("  Combined File Check — Applying Removals")
        self.log.emit("══════════════════════════════════════════")
        for fn in sorted(discard_file_nums, key=lambda x: float(x)):
            self.log.emit(f"  Removing File# {fn}")

        try:
            from file_io import remove_shots
            remove_shots(discard_file_nums, QC_DIR, REMOVED_DIR, log_fn=self.log.emit)
        except Exception:
            import traceback
            msg = traceback.format_exc()
            self.log.emit(f"\n  ERROR:\n{msg}")
            QMessageBox.critical(self, "Error", f"Removal failed:\n\n{msg}")
            return

        self.log.emit("══════════════════════════════════════════\n")
        n = len(discard_file_nums)
        QMessageBox.information(
            self, "Done",
            f"{n} shot{'s' if n > 1 else ''} removed. Re-checking…"
        )
        self._check()


# ── Main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("SeismicQC", "LogChecker")
        self.worker   = None
        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        self.setWindowTitle("Seismic Log QC Checker")
        self.setMinimumWidth(820)
        self.setMinimumHeight(640)

        root   = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)

        layout.addWidget(self._build_config_group())
        layout.addLayout(self._build_run_row())

        self.tabs = QTabWidget()

        # Tab 1 — log
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setFont(QFont("Courier New", 9))
        self.tabs.addTab(self.log_edit, "Log")

        # Tab 2 — station number check
        self.viz_tab = VizTab()
        self.viz_tab.log.connect(self._append_log)
        self.tabs.addTab(self.viz_tab, "Stn Num Check")

        # Tab 3 — combined file duplicate check
        self.combined_check_tab = CombinedCheckTab()
        self.combined_check_tab.log.connect(self._append_log)
        self.tabs.addTab(self.combined_check_tab, "Combined File Check")

        # Tab 4 — vibe performance
        self.vibe_tab = VibeTab()
        self.tabs.addTab(self.vibe_tab, "Vibe QC")

        layout.addWidget(self.tabs)

    def _build_config_group(self):
        group = QGroupBox("Configuration")
        vbox  = QVBoxLayout(group)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source Directory:"))
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("Folder containing the zip file…")
        src_row.addWidget(self.src_edit)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_source)
        src_row.addWidget(self.browse_btn)
        vbox.addLayout(src_row)

        zip_row = QHBoxLayout()
        zip_row.addWidget(QLabel("Zip File:"))
        self.zip_combo = QComboBox()
        self.zip_combo.setMinimumWidth(360)
        zip_row.addWidget(self.zip_combo)
        zip_row.addStretch()
        vbox.addLayout(zip_row)

        mv_row = QHBoxLayout()
        mv_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["zip", "folder", "both"])
        mv_row.addWidget(self.mode_combo)
        mv_row.addSpacing(24)
        mv_row.addWidget(QLabel("Vibes per Shot Point:"))
        self.vibes_spin = QSpinBox()
        self.vibes_spin.setMinimum(1)
        self.vibes_spin.setMaximum(99)
        self.vibes_spin.setValue(2)
        mv_row.addWidget(self.vibes_spin)
        mv_row.addStretch()
        vbox.addLayout(mv_row)

        return group

    def _build_run_row(self):
        row = QHBoxLayout()
        row.addStretch()
        self.run_btn = QPushButton("Run QC")
        self.run_btn.setFixedWidth(130)
        self.run_btn.clicked.connect(self._run)
        row.addWidget(self.run_btn)
        return row

    # ── Settings ───────────────────────────────────────────────────────────────

    def _load_settings(self):
        src = self.settings.value("last_source_dir", "")
        if src:
            self.src_edit.setText(src)
            self._populate_zips(src)

        last_zip = self.settings.value("last_zip", "")
        if last_zip:
            for i in range(self.zip_combo.count()):
                if self.zip_combo.itemData(i) == last_zip:
                    self.zip_combo.setCurrentIndex(i)
                    break

        mode = self.settings.value("mode", "zip")
        idx  = self.mode_combo.findText(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)

        self.vibes_spin.setValue(self.settings.value("vibes_per_point", 2, type=int))

    def _save_settings(self):
        self.settings.setValue("last_source_dir", self.src_edit.text())
        self.settings.setValue("last_zip",         self.zip_combo.currentData())
        self.settings.setValue("mode",             self.mode_combo.currentText())
        self.settings.setValue("vibes_per_point",  self.vibes_spin.value())

    # ── Slots ──────────────────────────────────────────────────────────────────

    def _browse_source(self):
        start = self.src_edit.text() or self.settings.value("last_source_dir", "")
        path  = QFileDialog.getExistingDirectory(self, "Select Source Directory", start)
        if path:
            self.src_edit.setText(path)
            self._populate_zips(path)

    @staticmethod
    def _fmt_size(path: Path) -> str:
        b = path.stat().st_size
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024:
                return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} TB"

    def _populate_zips(self, directory):
        previous = self.zip_combo.currentData()
        self.zip_combo.clear()
        zips = sorted(Path(directory).glob("*.zip"), reverse=True)
        for z in zips:
            self.zip_combo.addItem(f"{z.name}  ({self._fmt_size(z)})", userData=z.name)
        for i in range(self.zip_combo.count()):
            if self.zip_combo.itemData(i) == previous:
                self.zip_combo.setCurrentIndex(i)
                break

    def _run(self):
        source_dir = self.src_edit.text().strip()
        target_zip = (self.zip_combo.currentData() or "").strip()
        mode       = self.mode_combo.currentText()
        vibes      = self.vibes_spin.value()

        if not source_dir:
            self.log_edit.append("ERROR: Please select a source directory.")
            return
        if mode in ("zip", "both") and not target_zip:
            self.log_edit.append("ERROR: No zip file found in the selected directory.")
            return

        self._save_settings()
        self.log_edit.clear()
        self.tabs.setCurrentIndex(0)   # switch to Log tab while running
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running…")

        self.worker = Worker(source_dir, target_zip, mode, vibes)
        self.worker.log.connect(self._append_log)
        self.worker.voids_found.connect(self._on_voids_found)
        self.worker.duplicates_found.connect(self._on_duplicates_found)
        self.worker.results_ready.connect(self._on_results_ready)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _append_log(self, text):
        self.log_edit.append(text.rstrip())

    def _on_voids_found(self, shots):
        dlg = VoidReviewDialog(shots, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.worker.set_void_selections(dlg.get_reinstate_file_nums())
        else:
            self._append_log("  Void review cancelled — keeping all void shots removed.")
            self.worker.set_void_selections(set())

    def _on_duplicates_found(self, groups):
        dlg = DuplicateReviewDialog(groups, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.worker.set_selections(dlg.get_selections())
        else:
            self._append_log("  Duplicate review cancelled — keeping all shots.")
            self.worker.set_selections({})

    def _on_results_ready(self, obs_df, cog_df, pss_df, date_part):
        self.viz_tab.update_plots(obs_df, cog_df, date_part)
        self.vibe_tab.refresh(pss_df, cog_df, date_part)

    def _on_done(self, success, message):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run QC")
        if not success:
            self.log_edit.append(f"\n[FAILED] {message}")
        else:
            self.tabs.setCurrentIndex(1)   # switch to Visualisation tab on success

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
