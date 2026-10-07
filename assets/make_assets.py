"""Generates the SVG artwork in this folder: logo, README banners, the animated terminal demo
and the social preview. Run `python assets/make_assets.py` after editing it.

social-preview.png (the GitHub social card) is a 1280x640 render of social-preview.svg.
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

SANS = "'Segoe UI', Inter, -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif"
MONO = "ui-monospace, 'Cascadia Code', 'SF Mono', SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"


def mark(x: float, y: float, size: float, uid: str) -> str:
    """The Saccade mark: an eye whose pupil (a play button) has just jumped, leaving ghosts."""
    s = size / 128
    return f"""
  <g transform="translate({x} {y}) scale({s})">
    <defs>
      <linearGradient id="bg-{uid}" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#6366F1"/>
        <stop offset="0.55" stop-color="#3B82F6"/>
        <stop offset="1" stop-color="#06B6D4"/>
      </linearGradient>
      <linearGradient id="iris-{uid}" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#1E1B4B"/>
        <stop offset="1" stop-color="#0B1026"/>
      </linearGradient>
      <clipPath id="eye-{uid}">
        <path d="M14 64 C 36 30, 92 30, 114 64 C 92 98, 36 98, 14 64 Z"/>
      </clipPath>
    </defs>
    <rect width="128" height="128" rx="30" fill="url(#bg-{uid})"/>
    <path d="M14 64 C 36 30, 92 30, 114 64 C 92 98, 36 98, 14 64 Z" fill="#F8FAFC"/>
    <g clip-path="url(#eye-{uid})">
      <circle cx="44" cy="64" r="20" fill="#312E81" opacity="0.14"/>
      <circle cx="57" cy="64" r="20" fill="#312E81" opacity="0.28"/>
      <circle cx="72" cy="64" r="21" fill="url(#iris-{uid})"/>
      <path d="M66.5 53.5 L66.5 74.5 L84 64 Z" fill="#22D3EE" stroke="#22D3EE" stroke-width="2.5" stroke-linejoin="round"/>
    </g>
    <circle cx="80.5" cy="54.5" r="3.2" fill="#FFFFFF" opacity="0.9"/>
  </g>"""


def svg(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">\n'
        f"  <title>{escape(title)}</title>{body}\n</svg>\n"
    )


# ---------------------------------------------------------------- logo (icon only)
(OUT / "logo.svg").write_text(svg(128, 128, mark(0, 0, 128, "logo"), "Saccade"), encoding="utf-8")


# ---------------------------------------------------------------- banners
def timeline(x: float, y: float, w: float, dark: bool) -> str:
    """A video timeline: speech waveform bars, frame ticks and a few highlighted 'evidence' hits."""
    import math

    base = "#334155" if dark else "#CBD5E1"
    hit = "#22D3EE" if dark else "#0891B2"
    parts = []
    n = 96
    step = w / n
    for i in range(n):
        speech = (i // 9) % 3 != 2  # silence every third block
        h = 4 + (14 * abs(math.sin(i * 1.7)) + 6 * abs(math.sin(i * 0.37))) if speech else 2
        bx = x + i * step
        is_hit = i in (14, 15, 16, 47, 48, 77, 78, 79)
        parts.append(
            f'<rect x="{bx:.1f}" y="{y - h / 2:.1f}" width="{step * 0.55:.1f}" height="{h:.1f}" '
            f'rx="1.2" fill="{hit if is_hit else base}"/>'
        )
    for fx in (0.155, 0.5, 0.82):
        cx = x + w * fx
        parts.append(
            f'<path d="M{cx:.1f} {y - 26:.1f} l-5 -8 h10 z" fill="{hit}"/>'
            f'<line x1="{cx:.1f}" y1="{y - 24:.1f}" x2="{cx:.1f}" y2="{y + 18:.1f}" stroke="{hit}" '
            f'stroke-width="1.5" stroke-dasharray="2 3"/>'
        )
    labels = [(0.155, "[01:30]"), (0.5, "[13:05]"), (0.82, "[21:28]")]
    for fx, label in labels:
        parts.append(
            f'<text x="{x + w * fx:.1f}" y="{y + 36:.1f}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="12" fill="{hit}">{label}</text>'
        )
    return "\n  ".join(parts)


def banner(dark: bool) -> str:
    fg = "#F8FAFC" if dark else "#0F172A"
    sub = "#94A3B8" if dark else "#475569"
    body = mark(40, 40, 120, "b")
    body += f"""
  <text x="188" y="104" font-family="{SANS}" font-size="64" font-weight="700" letter-spacing="-1.5" fill="{fg}">saccade</text>
  <text x="191" y="142" font-family="{SANS}" font-size="21" fill="{sub}">Fast, local video context for LLMs</text>
  {timeline(540, 100, 350, dark)}"""
    return svg(920, 200, body, "Saccade: fast, local video context for LLMs")


(OUT / "banner-light.svg").write_text(banner(False), encoding="utf-8")
(OUT / "banner-dark.svg").write_text(banner(True), encoding="utf-8")


# ---------------------------------------------------------------- animated terminal demo
# Each entry: (seconds at which the line appears, [(text, colour), ...]).
GREY, WHITE, CYAN, GREEN, VIOLET, AMBER, DIM = (
    "#94A3B8", "#E2E8F0", "#22D3EE", "#4ADE80", "#A78BFA", "#FBBF24", "#64748B",
)
CODE = [
    [("import", VIOLET), (" saccade", WHITE)],
    [],
    [("video", WHITE), (" = ", GREY), ("saccade.Video", CYAN), ('(', GREY), ('"interview.mp4"', GREEN),
     (", llm=", GREY), ("saccade.azure", CYAN), ("(), device=", GREY), ('"cuda"', GREEN), (")", GREY)],
    [("answer", WHITE), (" = ", GREY), ("video.ask", CYAN), ("(", GREY),
     ('"How did the candidate handle the system design question?"', GREEN), (")", GREY)],
    [("print", VIOLET), ("(answer)", GREY)],
]
RUN = [
    (2.4, [("❯ ", GREEN), ("python ask.py", WHITE)]),
    (3.4, [("Inspecting media: interview.mp4", DIM)]),
    (3.9, [("Loading Whisper model 'small' (float16, cuda)", DIM)]),
    (4.5, [("Detected language: en (p=0.99)", DIM)]),
    (5.0, [("Transcribing 00:00:00–00:04:31 ", DIM), ("(17% of timeline)", DIM)]),
    (5.5, [("Transcribing 00:04:31–00:15:02 ", DIM), ("(57% of timeline)", DIM)]),
    (6.0, [("Completed: 412 segments, 52 frames · indexed 26 min of video in 9 s", GREY)]),
    (6.9, [("◆ ", VIOLET), ("Searching the transcript for 'system design'", WHITE)]),
    (7.7, [("◆ ", VIOLET), ("Reading the transcript 11:40–14:10", WHITE)]),
    (8.5, [("◆ ", VIOLET), ("Looking at 12:02, 13:05", WHITE)]),
    (9.3, [("◆ ", VIOLET), ("Zooming into 13:05", WHITE)]),
    (10.1, [("◆ ", VIOLET), ("Writing the answer", WHITE)]),
    (11.0, [("Confidently. At ", WHITE), ("[11:48]", CYAN), (" they sketched a cache in front of the", WHITE)]),
    (11.25, [("orders API and drew it on the whiteboard ", WHITE), ("[13:05]", CYAN),
             (", then explained", WHITE)]),
    (11.5, [("invalidation trade-offs when the interviewer pushed back ", WHITE), ("[13:40]", CYAN), (".", WHITE)]),
    (12.4, [("Tokens: 11,804 in (8,960 cached) + 640 out", AMBER)]),
]
CYCLE = 18.0
W, LINE, PAD = 900, 21, 28
code_top = 74
run_top = code_top + LINE * len(CODE) + 34
H = run_top + LINE * len(RUN) + 6


def spans(items: list[tuple[str, str]]) -> str:
    return "".join(f'<tspan fill="{c}">{escape(t)}</tspan>' for t, c in items)


css = [
    f"text{{font-family:{MONO};font-size:14px;white-space:pre}}",
    ".l{opacity:0}",
]
lines = []
for i, (at, items) in enumerate(RUN):
    show = at / CYCLE * 100
    css.append(
        f"@keyframes k{i}{{0%,{show:.2f}%{{opacity:0}}{show + 0.6:.2f}%,94%{{opacity:1}}98%,100%{{opacity:0}}}}"
        f".r{i}{{animation:k{i} {CYCLE}s linear infinite}}"
    )
    y = run_top + i * LINE
    lines.append(f'<text class="l r{i}" x="{PAD}" y="{y}" xml:space="preserve">{spans(items)}</text>')

# The code block "types" in: a cover rect slides off each code line in steps.
code_lines = []
type_end = 0.0
for i, items in enumerate(CODE):
    y = code_top + i * LINE
    code_lines.append(f'<text x="{PAD}" y="{y}" xml:space="preserve">{spans(items)}</text>')
    chars = sum(len(t) for t, _ in items)
    if not chars:
        continue
    dur = max(0.15, chars * 0.012)
    start, type_end = type_end, type_end + dur
    a, b = start / CYCLE * 100, type_end / CYCLE * 100
    css.append(
        f"@keyframes c{i}{{0%,{a:.2f}%{{transform:translateX(0)}}{b:.2f}%,100%{{transform:translateX({W}px)}}}}"
        f".c{i}{{animation:c{i} {CYCLE}s steps({chars}) infinite}}"
    )
    code_lines.append(
        f'<rect class="c{i}" x="{PAD - 2}" y="{y - 15}" width="{W}" height="{LINE}" fill="#0D1117"/>'
    )

body = f"""
  <defs>
    <linearGradient id="edge" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#6366F1"/>
      <stop offset="1" stop-color="#06B6D4"/>
    </linearGradient>
    <clipPath id="win"><rect width="{W}" height="{H}" rx="14"/></clipPath>
  </defs>
  <style>{"".join(css)}</style>
  <g clip-path="url(#win)">
    <rect width="{W}" height="{H}" fill="#0D1117"/>
    <rect width="{W}" height="40" fill="#161B22"/>
    <circle cx="22" cy="20" r="6" fill="#FF5F57"/>
    <circle cx="42" cy="20" r="6" fill="#FEBC2E"/>
    <circle cx="62" cy="20" r="6" fill="#28C840"/>
    <text x="{W / 2}" y="25" text-anchor="middle" fill="#8B949E" style="font-size:13px">ask.py — saccade</text>
    <text x="{PAD}" y="{code_top - 14}" fill="{DIM}" style="font-size:12px"># ask.py</text>
    {"".join(code_lines)}
    <line x1="{PAD}" y1="{run_top - 26}" x2="{W - PAD}" y2="{run_top - 26}" stroke="#21262D"/>
    {"".join(lines)}
  </g>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="14" fill="none" stroke="url(#edge)" stroke-opacity="0.55"/>"""
(OUT / "demo.svg").write_text(
    svg(W, H, body, "Animated terminal: indexing a 26-minute interview and asking a question about it"),
    encoding="utf-8",
)

# ---------------------------------------------------------------- social preview (1280x640)
social = f"""
  <defs>
    <radialGradient id="glow" cx="0.2" cy="0.1" r="1">
      <stop offset="0" stop-color="#1E1B4B"/>
      <stop offset="1" stop-color="#020617"/>
    </radialGradient>
  </defs>
  <rect width="1280" height="640" fill="url(#glow)"/>
  {mark(96, 150, 170, "s")}
  <text x="300" y="250" font-family="{SANS}" font-size="104" font-weight="700" letter-spacing="-2.5" fill="#F8FAFC">saccade</text>
  <text x="305" y="306" font-family="{SANS}" font-size="34" fill="#94A3B8">Fast, local video context for LLMs</text>
  <text x="100" y="420" font-family="{MONO}" font-size="27" fill="#E2E8F0" xml:space="preserve"><tspan fill="#94A3B8">video = </tspan><tspan fill="#22D3EE">saccade.Video</tspan><tspan fill="#94A3B8">(</tspan><tspan fill="#4ADE80">"interview.mp4"</tspan><tspan fill="#94A3B8">, llm=llm)</tspan></text>
  <text x="100" y="462" font-family="{MONO}" font-size="27" fill="#E2E8F0" xml:space="preserve"><tspan fill="#22D3EE">video.ask</tspan><tspan fill="#94A3B8">(</tspan><tspan fill="#4ADE80">"How did the interview go?"</tspan><tspan fill="#94A3B8">)</tspan></text>
  <text x="100" y="526" font-family="{SANS}" font-size="24" fill="#64748B">Local Whisper + VAD · representative frames · timestamped search · any LLM · CPU or GPU</text>
  <text x="100" y="584" font-family="{MONO}" font-size="24" fill="#22D3EE">pip install saccade-video</text>"""
(OUT / "social-preview.svg").write_text(svg(1280, 640, social, "Saccade"), encoding="utf-8")
print("written:", sorted(p.name for p in OUT.iterdir()))
