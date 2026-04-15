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


# ── Worker thread ──────────────────────────────────────────────────────────────

class Worker(QThread):
    log              = pyqtSignal(str)
    duplicates_found = pyqtSignal(list)
    results_ready    = pyqtSignal(object, object, str)   # obs_df, cog_df, date_part
    done             = pyqtSignal(bool, str)

    def __init__(self, source_dir, target_zip, mode, vibes_per_point):
        super().__init__()
        self.source_dir      = source_dir
        self.target_zip      = target_zip
        self.mode            = mode
        self.vibes_per_point = vibes_per_point
        self._resume_event   = threading.Event()
        self._selections     = {}

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
            obs_df, bad_obs_df, header_lines, obs_reasons = process_observer_log(obs_file)
            pss_df, bad_pss_df, pss_reasons               = process_pss_log(pss_file)
            cog_df, bad_cog_df, cog_reasons               = process_cog_log(cog_file)

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

            # Emit QC'd data for the Stn Num Check tab
            self.results_ready.emit(
                obs_df[['File#', 'Line', 'Station']].copy(),
                cog_df[['FF ID', 'Decoder Lat', 'Decoder Lon', 'Distance to Source Point']].copy()
                if {'FF ID', 'Decoder Lat', 'Decoder Lon', 'Distance to Source Point'}.issubset(cog_df.columns)
                else pd.DataFrame(),
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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # Shown when no data yet
        self._placeholder = QLabel("Run QC to populate charts.")
        self._placeholder.setAlignment(Qt.AlignCenter)
        outer.addWidget(self._placeholder)

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
                                      log_fn=self.log.emit)
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
        self.worker.duplicates_found.connect(self._on_duplicates_found)
        self.worker.results_ready.connect(self._on_results_ready)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _append_log(self, text):
        self.log_edit.append(text.rstrip())

    def _on_duplicates_found(self, groups):
        dlg = DuplicateReviewDialog(groups, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.worker.set_selections(dlg.get_selections())
        else:
            self._append_log("  Duplicate review cancelled — keeping all shots.")
            self.worker.set_selections({})

    def _on_results_ready(self, obs_df, cog_df, date_part):
        self.viz_tab.update_plots(obs_df, cog_df, date_part)

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
