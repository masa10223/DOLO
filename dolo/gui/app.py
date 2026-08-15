"""NiceGUI で構築する DOLO 推論ダッシュボード。"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from dolo.device import available_devices
from dolo.export import AVAILABLE_FORMATS
from dolo.tracking import TrackParams

from .config import (
    GUIPaths,
    ModelChoice,
    discover_default_model,
    validate_model_path,
)
from .jobs import InferenceConfig, JobManager, JobSnapshot, JobState
from .runtime import inspect_runtime
from .uploads import save_uploaded_video
from .video import VideoInfo, create_thumbnail, probe_video

ASSET_DIR = Path(__file__).with_name("assets")
ICON_PATH = ASSET_DIR / "dolo-icon.png"

CSS = r"""
:root {
  --dolo-ink: #173834;
  --dolo-muted: #647973;
  --dolo-paper: #eef3ee;
  --dolo-paper-deep: #dfe9e2;
  --dolo-card: rgba(255,253,247,.92);
  --dolo-line: rgba(25,78,72,.15);
  --dolo-green: #194e48;
  --dolo-green-deep: #103c37;
  --dolo-mint: #95aeab;
  --dolo-mint-soft: #c8d8d4;
  --dolo-ivory: #f8f8f5;
}
body { background: var(--dolo-paper); color: var(--dolo-ink); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
.q-layout, .q-page-container { background:
  radial-gradient(circle at 92% 2%, rgba(25,78,72,.15), transparent 30rem),
  radial-gradient(circle at 5% 44%, rgba(149,174,171,.18), transparent 34rem),
  linear-gradient(180deg,var(--dolo-paper) 0%,#f8f7f1 62%,#f2f5f0 100%); }
.dolo-header { background: rgba(16,60,55,.96) !important; color: var(--dolo-ivory) !important;
  border-bottom: 1px solid rgba(248,248,245,.12); backdrop-filter: blur(18px);
  box-shadow:0 10px 30px rgba(10,47,43,.15); }
.page-shell { width: min(1420px, calc(100vw - 40px)); margin: 0 auto; padding: 28px 0 72px; gap: 24px; }
.brand-mark { width: 44px; height: 44px; border-radius: 14px; overflow:hidden; flex:0 0 auto;
  border:1px solid rgba(248,248,245,.35); box-shadow: 0 8px 24px rgba(3,29,26,.32); }
.brand-mark img { width:100%; height:100%; display:block; object-fit:cover; }
.brand-subtitle { color:rgba(248,248,245,.62); font-size:9px; letter-spacing:.17em; }
.header-status { color:rgba(248,248,245,.7); font-size:10px; letter-spacing:.14em; font-weight:700; }
.hero-panel { min-height:310px; position:relative; overflow:hidden; padding:40px 42px; border-radius:32px;
  display:flex; align-items:center; isolation:isolate; color:var(--dolo-ivory);
  background:radial-gradient(circle at 84% 45%,rgba(149,174,171,.22),transparent 19rem),
    linear-gradient(112deg,#0d3935 0%,#194e48 62%,#245e57 100%);
  border:1px solid rgba(248,248,245,.12); box-shadow:0 24px 58px rgba(16,60,55,.19); }
.hero-panel:after { content:""; position:absolute; right:8%; bottom:-80px; width:280px; height:280px;
  border:1px solid rgba(248,248,245,.09); border-radius:50%; z-index:-1; }
.hero-copy-stack { position:relative; z-index:2; width:min(780px,calc(100% - 250px)); }
.hero-glyph { position:absolute; right:34px; top:50%; width:min(330px,28vw); aspect-ratio:1;
  transform:translateY(-50%) rotate(2deg); border-radius:24%; opacity:.48;
  box-shadow:0 20px 45px rgba(3,29,26,.22); }
.eyebrow { font-size: 11px; letter-spacing: .2em; font-weight: 800; text-transform: uppercase; color: var(--dolo-mint-soft); }
.hero-title { margin:0; font-size:clamp(68px,9vw,116px); line-height:.82; letter-spacing:-.065em; font-weight:900; }
.hero-expansion { font-size:clamp(21px,2.6vw,34px); line-height:1.1; letter-spacing:-.025em; font-weight:650; color:var(--dolo-ivory); }
.hero-copy { font-size: 16px; line-height: 1.72; color: rgba(248,248,245,.72); max-width: 650px; }
.surface { background: var(--dolo-card); border: 1px solid var(--dolo-line); border-radius: 26px;
  box-shadow: 0 18px 48px rgba(16,60,55,.08); }
.workspace-grid { display:grid; grid-template-columns: minmax(310px, .78fr) minmax(0, 1.5fr); gap:24px; width:100%; align-items:start; }
.sidebar-card { padding: 24px; gap: 18px; position: sticky; top: 92px; }
.preview-card { min-height: 360px; overflow:hidden; position:relative; background:var(--dolo-green-deep); color:var(--dolo-ivory); }
.preview-card .q-img { height: 100%; min-height: 350px; }
.preview-overlay { position:absolute; inset:auto 0 0; padding:58px 26px 24px; background:linear-gradient(transparent,rgba(8,38,34,.94)); z-index:2; }
.empty-preview { min-height:350px; width:100%; align-items:center; justify-content:center; text-align:center; padding:40px;
  background:radial-gradient(circle at 50% 42%,rgba(149,174,171,.2),transparent 17rem),
    linear-gradient(145deg,#103c37,#1d554f); }
.empty-icon-frame { width:138px; height:138px; padding:8px; margin-bottom:18px; border-radius:32px;
  border:1px solid rgba(248,248,245,.2); background:rgba(248,248,245,.05);
  box-shadow:0 20px 42px rgba(3,29,26,.25); }
.empty-icon-frame img { display:block; width:100%; height:100%; object-fit:cover; border-radius:25px; }
.section-title { color:var(--dolo-ink); font-size: 22px; font-weight: 780; letter-spacing:-.025em; }
.micro-label { color:#4b7570; font-size:11px; font-weight:780; letter-spacing:.13em; text-transform:uppercase; }
.dolo-muted { color:var(--dolo-muted); }
.hairline { height:1px; width:100%; background:var(--dolo-line); }
.primary-cta { border-radius:15px !important; min-height:48px; padding:0 20px; font-weight:760; letter-spacing:.01em; }
.status-dot { width:8px;height:8px;border-radius:50%;display:inline-block;background:#b7c8c6;box-shadow:0 0 0 5px rgba(183,200,198,.14); }
.format-tile { border:1px solid var(--dolo-line); border-radius:15px; padding:9px 12px; background:rgba(223,233,226,.38); transition:border-color .2s,background .2s; }
.format-tile:hover { border-color:rgba(25,78,72,.3); background:rgba(223,233,226,.62); }
.run-card { padding:24px; gap:20px; }
.metric-grid { display:grid; grid-template-columns:repeat(4,minmax(120px,1fr)); gap:12px; width:100%; }
.metric-card { border:1px solid var(--dolo-line); border-radius:18px; padding:16px; background:rgba(223,233,226,.42); }
.metric-value { font-size:27px; font-weight:820; letter-spacing:-.04em; }
.log-box textarea { font-family:ui-monospace,SFMono-Regular,Menlo,monospace !important; font-size:11px !important; line-height:1.55 !important; }
.analysis-card { border:1px solid var(--dolo-line) !important; border-radius:18px !important; box-shadow:none !important; background:rgba(255,255,255,.56) !important; }
.output-card { border:1px solid var(--dolo-line); border-radius:16px; padding:14px 16px; background:rgba(255,255,255,.64); }
.video-preview { border:1px solid var(--dolo-line); border-radius:18px; overflow:hidden; background:#0c1f1d; }
.video-preview video { width:100%; max-height:620px; display:block; background:#0c1f1d; }
.picker-dialog { width:min(760px,94vw); max-width:min(760px,94vw) !important; border-radius:26px !important; padding:0 !important; overflow:hidden; background:#fffdf7 !important; }
.picker-head { padding:24px 26px 14px; }
.picker-body { padding:8px 26px 26px; }
.results-mark { color:var(--dolo-mint); opacity:.45; }
.q-field--outlined .q-field__control { border-radius:13px; background:rgba(255,255,255,.55); }
.q-field--outlined .q-field__control:before { border-color:rgba(25,78,72,.22); }
.q-field--outlined.q-field--focused .q-field__control:after { border-color:var(--dolo-green); }
.q-tabs { color:#5c7772; }
.q-tab--active { color:var(--dolo-green); }
.q-expansion-item { border-radius:14px; color:var(--dolo-ink); }
.q-btn.bg-primary { box-shadow:0 10px 22px rgba(25,78,72,.2); }
.q-btn { text-transform:none; }
.research-credit { width:100%; display:grid; grid-template-columns:minmax(150px,.32fr) minmax(0,1.68fr); gap:22px;
  padding:24px 28px; border:1px solid var(--dolo-line); border-radius:22px; background:rgba(255,253,247,.64);
  color:var(--dolo-muted); font-size:12px; line-height:1.7; }
.research-credit__label { color:#4b7570; font-size:10px; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.research-credit strong { color:var(--dolo-ink); }
.research-credit a { color:var(--dolo-green); font-weight:700; text-decoration:none; }
.research-credit a:hover { text-decoration:underline; }
@media (max-width: 900px) {
 .workspace-grid { grid-template-columns:1fr; }
 .sidebar-card { position:static; }
 .metric-grid { grid-template-columns:repeat(2,1fr); }
 .page-shell { width:min(100% - 24px,1480px); padding-top:22px; }
 .hero-panel { min-height:300px; padding:30px 26px; }
 .hero-copy-stack { width:100%; }
 .hero-glyph { right:-52px; width:260px; opacity:.2; }
 .research-credit { grid-template-columns:1fr; gap:8px; }
}
@media (max-width: 520px) {
 .header-status-group { display:none !important; }
}
"""


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _state_label(state: JobState) -> str:
    return {
        JobState.QUEUED: "待機中",
        JobState.RUNNING: "推論中",
        JobState.COMPLETE: "完了",
        JobState.FAILED: "失敗",
        JobState.CANCELLED: "中断",
    }[state]


@dataclass
class _Selection:
    path: Path | None = None
    info: VideoInfo | None = None
    thumbnail: Path | None = None


class Dashboard:
    def __init__(self, paths: GUIPaths, manager: JobManager, model: ModelChoice) -> None:
        self.paths = paths
        self.manager = manager
        self.model = model
        self.selection = _Selection()
        self.job_id: str | None = None
        self.rendered_terminal_job: str | None = None

    def build(self) -> None:
        from nicegui import ui

        ui.add_css(CSS)
        ui.colors(
            primary="#194e48",
            secondary="#95aeab",
            accent="#c8d8d4",
            dark="#103c37",
            positive="#4b756d",
            warning="#9a753b",
            negative="#a54b46",
        )

        with ui.header(elevated=False).classes("dolo-header h-[68px] items-center px-5 md:px-9"):
            with ui.row().classes("items-center gap-3"):
                ui.html(
                    '<div class="brand-mark">'
                    '<img src="/dolo-assets/dolo-icon.png" alt="DOLO icon">'
                    "</div>"
                )
                with ui.column().classes("gap-0"):
                    ui.label("DOLO").classes("font-black tracking-tight text-lg leading-5")
                    ui.label("DROSOPHILA MOTION LAB").classes("brand-subtitle")
            ui.space()
            with ui.row().classes("header-status-group items-center gap-3"):
                ui.html('<span class="status-dot"></span>')
                ui.label("LOCAL INFERENCE").classes("header-status")

        self._build_video_dialog()

        with ui.column().classes("page-shell"):
            with ui.element("section").classes("hero-panel w-full"):
                with ui.column().classes("hero-copy-stack gap-4"):
                    ui.label("MARKERLESS MULTI-LARVA TRACKING").classes("eyebrow")
                    ui.html('<h1 class="hero-title">DOLO</h1>')
                    ui.html(
                        '<div class="hero-expansion"><em>Drosophila</em> tracking with YOLO</div>'
                    )
                    ui.label(
                        "Markerless pose estimation, identity tracking, and behavioral analysis for "
                        "Drosophila larvae—from video input to reproducible trajectories and annotated outputs."
                    ).classes("hero-copy")
                ui.html('<img class="hero-glyph" src="/dolo-assets/dolo-icon.png" alt="">')

            with ui.element("div").classes("workspace-grid"):
                with ui.column().classes("surface sidebar-card"):
                    self._build_controls()
                with ui.column().classes("gap-6 min-w-0"):
                    self._build_preview()
                    self.results_container = ui.column().classes("w-full gap-6")
                    self._render_empty_results()

            ui.html(
                '<footer class="research-credit">'
                '<div class="research-credit__label">Project provenance</div>'
                "<div><strong>DOLO was designed and implemented by "
                '<a href="mailto:mtsutsumi@nagoya-u.jp">Masato Tsutsumi</a></strong> for the study '
                "<cite>“Chemosensory input suppresses cannibalism by stabilizing social feeding boundaries "
                "in Drosophila larvae”</cite> by Nagisa Matsuda-Watanabe, Masato Tsutsumi, Misako Okumura, "
                "and Takahiro Chihara. © 2024–2026 Masato Tsutsumi. AGPL-3.0-or-later.</div>"
                "</footer>"
            )

        ui.timer(0.5, self._poll_job)

    def _build_video_dialog(self) -> None:
        from nicegui import ui

        with ui.dialog() as self.video_dialog, ui.card().classes("picker-dialog"):
            with ui.row().classes("picker-head w-full items-start"):
                with ui.column().classes("gap-1"):
                    ui.label("動画を選択").classes("section-title")
                    ui.label("ローカルからアップロード、またはサーバー上のパスを指定").classes(
                        "text-sm dolo-muted"
                    )
                ui.space()
                ui.button(icon="close", on_click=self.video_dialog.close).props("flat round")

            with ui.tabs().classes("w-full px-5") as tabs:
                upload_tab = ui.tab("アップロード", icon="upload_file")
                path_tab = ui.tab("サーバーパス", icon="folder_open")
            with ui.tab_panels(tabs, value=upload_tab).classes("picker-body w-full bg-transparent"):
                with ui.tab_panel(upload_tab).classes("px-0"):
                    self.uploader = (
                        ui.upload(
                            label="動画をここへドロップ",
                            on_upload=self._handle_upload,
                            on_rejected=lambda: ui.notify(
                                "動画を受け付けられませんでした。形式またはファイルを確認してください",
                                type="negative",
                            ),
                            auto_upload=True,
                            multiple=False,
                        )
                        .props(
                            'accept="video/*,.avi,.m4v,.mkv,.mov,.mp4,.mpeg,.mpg,.webm" flat bordered color=primary'
                        )
                        .classes("w-full min-h-[180px]")
                    )
                    ui.label(
                        "大きな動画はアップロード完了までこの画面を閉じないでください。"
                    ).classes("text-xs dolo-muted mt-3")
                with ui.tab_panel(path_tab).classes("px-0"):
                    self.server_path_input = (
                        ui.input("動画の絶対パス", placeholder="/data/videos/experiment_01.mov")
                        .props("outlined clearable")
                        .classes("w-full")
                    )
                    ui.label(
                        "GPUサーバーで起動している場合は、サーバー側から見えるパスを指定します。"
                    ).classes("text-xs dolo-muted")
                    ui.button(
                        "この動画を使う", icon="arrow_forward", on_click=self._handle_server_path
                    ).props("unelevated color=primary").classes("primary-cta mt-5")

    def _build_controls(self) -> None:
        from nicegui import ui

        with ui.row().classes("w-full items-center"):
            with ui.column().classes("gap-0"):
                ui.label("Inference setup").classes("micro-label")
                ui.label("推論の準備").classes("section-title")
            ui.space()
            ui.button(icon="add", on_click=self.video_dialog.open).props(
                "round unelevated color=primary"
            )

        ui.html('<div class="hairline"></div>')
        self.selected_name = ui.label("動画が未選択です").classes("font-bold text-[15px] break-all")
        self.selected_meta = ui.label("＋ から動画を追加してください").classes("text-xs dolo-muted")
        ui.button("動画を選ぶ", icon="video_file", on_click=self.video_dialog.open).props(
            "outline color=primary"
        ).classes("w-full primary-cta")

        ui.label("MODEL & DEVICE").classes("micro-label mt-2")
        self.model_input = (
            ui.input(
                "モデル重み",
                value=str(self.model.path) if self.model.available else "",
                placeholder="/path/to/best.pt",
            )
            .props("outlined dense")
            .classes("w-full")
        )
        if self.model.available:
            self.model_hint = ui.label(f"default · {self.model.path.name}").classes(
                "text-xs text-emerald-700"
            )
        else:
            self.model_hint = ui.label(self.model.warning or "モデルが必要です").classes(
                "text-xs text-orange-700"
            )

        devices = ["auto", *available_devices()]
        devices = list(dict.fromkeys(devices))
        self.device_select = (
            ui.select(devices, value="auto", label="計算デバイス")
            .props("outlined dense options-dense")
            .classes("w-full")
        )

        ui.label("OUTPUTS").classes("micro-label mt-2")
        self.format_checks = {}
        defaults = {"csv", "json", "mov"}
        for key, (label, description, _ext) in AVAILABLE_FORMATS.items():
            with ui.row().classes("format-tile w-full items-start gap-2 no-wrap"):
                checkbox = ui.checkbox(value=key in defaults).props("dense color=primary")
                with ui.column().classes("gap-0 min-w-0"):
                    ui.label(label).classes("text-sm font-bold")
                    ui.label(description).classes("text-[10px] leading-4 dolo-muted")
            self.format_checks[key] = checkbox

        with ui.expansion("詳細設定", icon="tune").classes("w-full"):
            with ui.column().classes("w-full gap-3 pt-3"):
                self.max_ids = (
                    ui.number("個体数の上限", value=5, min=1, max=100, step=1)
                    .props("outlined dense")
                    .classes("w-full")
                )
                with ui.row().classes("w-full gap-3"):
                    self.confidence = (
                        ui.number(
                            "検出 confidence",
                            value=0.001,
                            min=0.0001,
                            max=1,
                            step=0.001,
                            format="%.4f",
                        )
                        .props("outlined dense")
                        .classes("flex-1")
                    )
                    self.iou = (
                        ui.number("IoU", value=0.45, min=0.05, max=0.95, step=0.05)
                        .props("outlined dense")
                        .classes("flex-1")
                    )
                with ui.row().classes("w-full gap-3"):
                    self.start_frame = (
                        ui.number("開始 frame", value=0, min=0, step=1)
                        .props("outlined dense")
                        .classes("flex-1")
                    )
                    self.end_frame = (
                        ui.number("終了 frame", value=None, min=1, step=1)
                        .props("outlined dense clearable")
                        .classes("flex-1")
                    )
                with ui.row().classes("w-full gap-3"):
                    self.max_age = (
                        ui.number("見失い許容", value=15, min=1, step=1)
                        .props("outlined dense")
                        .classes("flex-1")
                    )
                    self.frame_skip = (
                        ui.number("frame skip", value=1, min=1, step=1)
                        .props("outlined dense")
                        .classes("flex-1")
                    )
                self.trail_frames = (
                    ui.number(
                        "中心軌跡を残す frame 数",
                        value=30,
                        min=0,
                        max=100000,
                        step=1,
                    )
                    .props("outlined dense")
                    .classes("w-full")
                )
                ui.label(
                    "中心軌跡動画だけに適用します。0にすると最初からの全軌跡を残します。"
                ).classes("text-[10px] leading-4 dolo-muted -mt-2")
                self.output_input = (
                    ui.input("出力先", value=str(self.paths.runs))
                    .props("outlined dense")
                    .classes("w-full")
                )

        self.start_button = (
            ui.button("推論を開始", icon="play_arrow", on_click=self._start_job)
            .props("unelevated color=primary")
            .classes("w-full primary-cta")
        )
        runtime = inspect_runtime()
        if not runtime.inference_ready:
            missing = ", ".join(runtime.missing_inference)
            ui.label(f"推論依存が不足: {missing} — dolo doctor で確認").classes(
                "text-[11px] text-orange-700"
            )

    def _build_preview(self) -> None:
        from nicegui import ui

        with ui.card().classes("surface preview-card w-full p-0"):
            self.empty_preview = ui.html(
                '<div class="empty-preview flex flex-col">'
                '<div class="empty-icon-frame">'
                '<img src="/dolo-assets/dolo-icon.png" alt="DOLO larval interaction icon">'
                "</div>"
                '<div style="font-weight:800;font-size:21px">Drop a motion study</div>'
                '<div style="opacity:.65;font-size:12px;margin-top:7px">MOV · MP4 · AVI · MKV · WEBM</div>'
                "</div>"
            ).classes("w-full")
            self.preview_image = ui.image().classes("w-full object-cover").props("fit=cover")
            self.preview_image.visible = False
            with ui.column().classes("preview-overlay gap-1") as self.preview_overlay:
                self.preview_overlay.visible = False
                self.preview_title = ui.label().classes("text-xl font-black")
                self.preview_detail = ui.label().classes("text-xs opacity-70")

    def _render_empty_results(self) -> None:
        from nicegui import ui

        self.results_container.clear()
        with self.results_container:
            with ui.card().classes("surface run-card w-full"):
                with ui.row().classes("w-full items-center"):
                    with ui.column().classes("gap-1"):
                        ui.label("RESULTS").classes("micro-label")
                        ui.label("結果はここに集まります").classes("section-title")
                    ui.space()
                    ui.icon("query_stats", size="34px").classes("results-mark")
                ui.label(
                    "推論を開始すると、進捗・ログ・個体別メトリクス・出力ファイルがこのパネルに表示されます。"
                ).classes("text-sm dolo-muted")

    async def _handle_upload(self, event) -> None:
        from nicegui import run, ui

        file = getattr(event, "file", None)
        if file is None:
            ui.notify("このNiceGUIバージョンのアップロードAPIに対応していません", type="negative")
            return
        destination = None
        try:
            destination = await save_uploaded_video(file, self.paths.uploads)
            await self._select_video(destination, run)
        except Exception as exc:  # noqa: BLE001 - UI境界
            if destination is not None:
                destination.unlink(missing_ok=True)
            ui.notify(f"動画を受け付けられません: {exc}", type="negative", timeout=8000)

    async def _handle_server_path(self) -> None:
        from nicegui import run, ui

        value = (self.server_path_input.value or "").strip()
        if not value:
            ui.notify("動画のパスを入力してください", type="warning")
            return
        try:
            await self._select_video(Path(value), run)
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative", timeout=8000)

    async def _select_video(self, path: Path, runner) -> None:
        from nicegui import ui

        info = await runner.io_bound(probe_video, path)
        thumb = self.paths.thumbnails / f"{uuid.uuid4().hex}.jpg"
        await runner.io_bound(create_thumbnail, info, thumb)
        self.selection = _Selection(path=info.path, info=info, thumbnail=thumb)

        self.selected_name.text = info.path.name
        self.selected_meta.text = f"{info.resolution} · {info.fps:.2f} fps · {info.frames:,} frames · {info.duration_label}"
        self.end_frame.value = info.frames or None
        self.empty_preview.visible = False
        self.preview_image.set_source(self._media_url(thumb))
        self.preview_image.visible = True
        self.preview_overlay.visible = True
        self.preview_title.text = info.path.name
        self.preview_detail.text = f"{info.resolution}   {info.duration_label}   {_human_bytes(info.size_bytes)}   {info.codec}"
        self.video_dialog.close()
        ui.notify("動画を読み込みました", type="positive")

    def _read_config(self) -> InferenceConfig:
        if self.selection.path is None:
            raise ValueError("先に動画を選んでください")
        model = validate_model_path(self.model_input.value or "")
        formats = frozenset(key for key, box in self.format_checks.items() if box.value)
        params = TrackParams(
            max_ids=int(self.max_ids.value or 5),
            conf_thres=float(self.confidence.value or 0.001),
            iou_thres=float(self.iou.value or 0.45),
            max_age=int(self.max_age.value or 15),
            frame_skip=int(self.frame_skip.value or 1),
        )
        return InferenceConfig(
            video_path=self.selection.path,
            model_path=model,
            output_root=Path(self.output_input.value or self.paths.runs),
            formats=formats,
            params=params,
            device=str(self.device_select.value or "auto"),
            start_frame=int(self.start_frame.value or 0),
            end_frame=int(self.end_frame.value) if self.end_frame.value is not None else None,
            renderer="fast",
            trail_frames=int(self.trail_frames.value or 0),
        )

    def _start_job(self) -> None:
        from nicegui import ui

        try:
            snapshot = self.manager.submit(self._read_config())
        except Exception as exc:  # noqa: BLE001
            ui.notify(str(exc), type="negative", timeout=8000)
            return
        self.job_id = snapshot.id
        self.rendered_terminal_job = None
        self.start_button.disable()
        self._render_running(snapshot)
        ui.notify("推論ジョブを開始しました", type="positive")

    def _render_running(self, snapshot: JobSnapshot) -> None:
        from nicegui import ui

        self.results_container.clear()
        with self.results_container:
            with ui.card().classes("surface run-card w-full"):
                with ui.row().classes("w-full items-center"):
                    with ui.column().classes("gap-0"):
                        ui.label("ACTIVE RUN").classes("micro-label")
                        self.run_status = ui.label(_state_label(snapshot.state)).classes(
                            "section-title"
                        )
                    ui.space()
                    self.cancel_button = ui.button(
                        "中断", icon="stop", on_click=self._cancel_job
                    ).props("outline color=negative")
                self.progress = (
                    ui.linear_progress(value=snapshot.progress, show_value=False)
                    .props("rounded size=12px color=primary track-color=grey-3")
                    .classes("w-full")
                )
                self.progress_label = ui.label("モデルを準備しています…").classes(
                    "text-xs dolo-muted"
                )
                self.log_area = (
                    ui.textarea("ログ", value="\n".join(snapshot.logs))
                    .props("outlined readonly autogrow")
                    .classes("w-full log-box")
                )

    def _poll_job(self) -> None:
        if self.job_id is None:
            return
        try:
            snapshot = self.manager.snapshot(self.job_id)
        except KeyError:
            return
        if snapshot.state.terminal:
            if self.rendered_terminal_job != snapshot.id:
                self.rendered_terminal_job = snapshot.id
                self.start_button.enable()
                self._render_terminal(snapshot)
            return
        if not hasattr(self, "progress"):
            return
        self.run_status.text = _state_label(snapshot.state)
        self.progress.value = snapshot.progress
        if snapshot.total:
            self.progress_label.text = (
                f"{snapshot.done:,} / {snapshot.total:,} frames  ·  {snapshot.progress * 100:.1f}%"
            )
        else:
            self.progress_label.text = "モデルと動画を準備しています…"
        self.log_area.value = "\n".join(snapshot.logs)
        self.log_area.update()

    def _cancel_job(self) -> None:
        from nicegui import ui

        if self.job_id and self.manager.cancel(self.job_id):
            self.cancel_button.disable()
            ui.notify("中断を要求しました", type="warning")

    def _render_terminal(self, snapshot: JobSnapshot) -> None:
        from nicegui import ui

        self.results_container.clear()
        with self.results_container:
            with ui.card().classes("surface run-card w-full"):
                with ui.row().classes("w-full items-center"):
                    with ui.column().classes("gap-0"):
                        ui.label(
                            "RUN COMPLETE" if snapshot.state == JobState.COMPLETE else "RUN REPORT"
                        ).classes("micro-label")
                        ui.label(_state_label(snapshot.state)).classes("section-title")
                    ui.space()
                    color = "positive" if snapshot.state == JobState.COMPLETE else "negative"
                    icon = "check_circle" if snapshot.state == JobState.COMPLETE else "error"
                    ui.icon(icon, color=color, size="36px")

                if snapshot.error:
                    ui.label(snapshot.error).classes(
                        "text-sm text-red-700 bg-red-50 rounded-xl p-4 w-full"
                    )

                summary = snapshot.summary
                metrics = snapshot.metrics
                cards = [
                    (
                        "Frames",
                        f"{summary.frames_processed:,}" if summary else f"{snapshot.done:,}",
                    ),
                    ("Tracked IDs", f"{len(summary.ids_seen):,}" if summary else "—"),
                    ("Rows", f"{summary.rows_written:,}" if summary else "—"),
                    ("Elapsed", f"{summary.elapsed_sec:.1f}s" if summary else "—"),
                ]
                with ui.element("div").classes("metric-grid"):
                    for label, value in cards:
                        with ui.column().classes("metric-card gap-1"):
                            ui.label(label).classes("micro-label")
                            ui.label(value).classes("metric-value")

                if metrics and metrics.ids:
                    with ui.row().classes("w-full gap-5 items-stretch"):
                        with ui.card().classes("analysis-card flex-1 min-w-[300px]"):
                            ui.label("個体別の総移動量").classes("font-bold")
                            ui.echart(
                                {
                                    "grid": {"left": 52, "right": 18, "top": 24, "bottom": 34},
                                    "tooltip": {"trigger": "axis"},
                                    "xAxis": {
                                        "type": "category",
                                        "data": [f"ID {m.track_id}" for m in metrics.ids],
                                        "axisLine": {"lineStyle": {"color": "#aab7b4"}},
                                    },
                                    "yAxis": {
                                        "type": "value",
                                        "name": "px",
                                        "splitLine": {"lineStyle": {"color": "#e7ecea"}},
                                    },
                                    "series": [
                                        {
                                            "type": "bar",
                                            "data": [
                                                round(m.total_distance_px, 2) for m in metrics.ids
                                            ],
                                            "itemStyle": {
                                                "color": "#194e48",
                                                "borderRadius": [6, 6, 0, 0],
                                            },
                                        }
                                    ],
                                }
                            ).classes("w-full h-[280px]")

                    columns = [
                        {"name": "id", "label": "ID", "field": "id", "align": "left"},
                        {"name": "frames", "label": "Frames", "field": "frames", "sortable": True},
                        {
                            "name": "coverage",
                            "label": "Coverage",
                            "field": "coverage",
                            "sortable": True,
                        },
                        {
                            "name": "distance",
                            "label": "Distance (px)",
                            "field": "distance",
                            "sortable": True,
                        },
                        {
                            "name": "confidence",
                            "label": "Confidence",
                            "field": "confidence",
                            "sortable": True,
                        },
                    ]
                    rows = [
                        {
                            "id": item.track_id,
                            "frames": item.visible_frames,
                            "coverage": f"{item.coverage * 100:.1f}%",
                            "distance": f"{item.total_distance_px:.1f}",
                            "confidence": f"{item.mean_confidence:.3f}",
                        }
                        for item in metrics.ids
                    ]
                    ui.table(columns=columns, rows=rows, row_key="id", pagination=10).props(
                        "flat bordered"
                    ).classes("w-full")

                existing = [path for path in snapshot.outputs if path.exists()]
                playable = [
                    path
                    for path in existing
                    if path.suffix.lower() == ".mp4" and self._is_media_path(path)
                ]
                if playable:
                    ui.label("VIDEO PREVIEW").classes("micro-label mt-2")
                    ui.label("出力動画をこの画面で再生できます").classes("font-bold")
                    with ui.tabs().classes("w-full") as video_tabs:
                        tab_by_path = {
                            path: ui.tab(self._video_label(path), icon="play_circle")
                            for path in playable
                        }
                    with ui.tab_panels(video_tabs, value=tab_by_path[playable[0]]).classes(
                        "video-preview w-full p-0"
                    ):
                        for path in playable:
                            with ui.tab_panel(tab_by_path[path]).classes("p-0"):
                                ui.video(
                                    self._media_url(path),
                                    controls=True,
                                    autoplay=False,
                                    muted=True,
                                ).classes("w-full")

                if existing:
                    ui.label("OUTPUT FILES").classes("micro-label mt-2")
                    for path in existing:
                        with ui.row().classes("output-card w-full items-center no-wrap"):
                            ui.icon(self._file_icon(path), size="24px").classes("text-primary")
                            with ui.column().classes("gap-0 min-w-0"):
                                ui.label(path.name).classes("font-bold text-sm break-all")
                                ui.label(_human_bytes(path.stat().st_size)).classes(
                                    "text-[10px] dolo-muted"
                                )
                            ui.space()
                            ui.button(
                                icon="download", on_click=lambda p=path: ui.download(str(p))
                            ).props("flat round color=primary")
                    ui.label(f"保存先: {snapshot.run_dir}").classes(
                        "text-[11px] dolo-muted break-all"
                    )

                if snapshot.logs:
                    with ui.expansion("実行ログ", icon="terminal").classes("w-full"):
                        ui.textarea(value="\n".join(snapshot.logs)).props(
                            "outlined readonly autogrow"
                        ).classes("w-full log-box")

    def _media_url(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.paths.root.resolve())
        return "/dolo-media/" + "/".join(quote(part) for part in relative.parts)

    def _is_media_path(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.paths.root.resolve())
        except ValueError:
            return False
        return True

    @staticmethod
    def _video_label(path: Path) -> str:
        name = path.stem
        if name.endswith("_center_track"):
            return "中心軌跡"
        if name.endswith("_angle"):
            return "角度"
        if name.endswith("_pose"):
            return "姿勢"
        return name

    @staticmethod
    def _file_icon(path: Path) -> str:
        return {
            ".csv": "table_view",
            ".json": "data_object",
            ".jsonl": "data_object",
            ".mov": "movie",
            ".mp4": "movie",
            ".gif": "gif_box",
        }.get(path.suffix.lower(), "description")


def create_gui(
    *,
    data_dir: str | Path | None = None,
    model: str | Path | None = None,
    manager: JobManager | None = None,
) -> JobManager:
    """NiceGUI のルートページとメディア配信を登録する。"""
    try:
        from nicegui import app, ui
    except ImportError as exc:  # pragma: no cover - CLIでメッセージを確認する
        raise ImportError(
            "GUIにはNiceGUIが必要です。`pip install -e '.[gui]'` を実行してください。"
        ) from exc

    paths = GUIPaths.from_environment(data_dir).ensure()
    ultralytics_dir = paths.root / "ultralytics"
    ultralytics_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir = paths.root / "matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ultralytics_dir))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    choice = discover_default_model(model, data_root=paths.root)
    manager = manager or JobManager(max_workers=int(os.environ.get("DOLO_MAX_JOBS", "1")))

    app.add_media_files("/dolo-media", str(paths.root))
    app.add_static_files("/dolo-assets", str(ASSET_DIR))

    @app.get("/api/health")
    def health():
        runtime = inspect_runtime()
        return {
            "status": "ok",
            "inference_ready": runtime.inference_ready,
            "model_ready": choice.available,
            "data_dir": str(paths.root),
        }

    @ui.page("/")
    def index():
        Dashboard(paths, manager, choice).build()

    return manager


def run_gui(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    data_dir: str | Path | None = None,
    model: str | Path | None = None,
    reload: bool = False,
    show: bool = True,
) -> None:
    """GUI サーバーを起動する。"""
    try:
        from nicegui import ui
    except ImportError as exc:
        raise SystemExit(
            "NiceGUI がありません。`pip install -e '.[gui]'` を実行してください。"
        ) from exc

    manager = create_gui(data_dir=data_dir, model=model)
    try:
        try:
            ui.run(
                host=host,
                port=port,
                title="DOLO — Drosophila Motion Lab",
                favicon=str(ICON_PATH),
                reload=reload,
                show=show,
                language="ja",
            )
        except KeyboardInterrupt:  # Ctrl-C は正常な終了操作
            pass
    finally:
        manager.shutdown(wait=False)
