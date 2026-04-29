import re

with open('d:/HybridRAG_Sanius/frontend/app.py', encoding='utf-8', errors='replace') as f:
    content = f.read()

# ── 1. Replace the CSS welcome-screen block with theme-aware version ──────────
OLD_CSS = """    /* suppress any Streamlit focus ring on injected HTML */
    .stMarkdown p, .stMarkdown div { outline: none !important; }

    .welcome-heading {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.2rem;
    }
    .welcome-sub {
        font-size: 0.87rem;
        color: #94a3b8;
        margin-bottom: 1.3rem;
    }
    .ds-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.6rem;
        height: 100%;
    }
    .ds-card-title {
        font-weight: 700;
        font-size: 0.92rem;
        color: #93c5fd;
        margin-bottom: 0.35rem;
    }
    .ds-card-desc {
        font-size: 0.82rem;
        color: #cbd5e1;
        line-height: 1.5;
    }
    .starters-heading {
        font-size: 0.92rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1.4rem 0 0.6rem;
    }
    /* Card + ask-button wrapper \u2014 all same height, button pinned to bottom */
    .starter-wrap {
        display: flex;
        flex-direction: column;
        height: 100%;
        min-height: 165px;
        border-radius: 12px;
        border: 1.5px solid;
        overflow: hidden;
    }
    .starter-easy   { background:rgba(22,163,74,0.12);  border-color:rgba(134,239,172,0.4); }
    .starter-medium { background:rgba(217,119,6,0.12);  border-color:rgba(252,211,77,0.4); }
    .starter-hard   { background:rgba(220,38,38,0.12);  border-color:rgba(252,165,165,0.4); }
    .starter-body {
        flex: 1;
        padding: 0.85rem 1rem 0.75rem;
    }
    .starter-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .label-easy   { color: #4ade80; }
    .label-medium { color: #fcd34d; }
    .label-hard   { color: #f87171; }
    .starter-text {
        font-size: 0.87rem;
        color: #e2e8f0;
        font-weight: 500;
        line-height: 1.45;
    }
    .starter-btn {
        display: block;
        width: 100%;
        padding: 0.55rem 1rem;
        background: rgba(255,255,255,0.07);
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        color: #e2e8f0;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        cursor: pointer;
        letter-spacing: 0.2px;
    }
    .starter-btn:hover { background: rgba(255,255,255,0.13); }"""

NEW_CSS = """    /* suppress Streamlit focus ring on injected HTML */
    .stMarkdown p, .stMarkdown div { outline: none !important; }

    /* Theme-aware: Streamlit CSS vars work in BOTH dark and light mode */
    .welcome-heading {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.3rem;
    }
    .welcome-sub {
        font-size: 0.87rem;
        color: var(--text-color);
        opacity: 0.6;
        margin-bottom: 1.3rem;
    }
    .ds-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.6rem;
        height: 100%;
    }
    .ds-card-title {
        font-weight: 700;
        font-size: 0.92rem;
        color: var(--primary-color);
        margin-bottom: 0.35rem;
    }
    .ds-card-desc {
        font-size: 0.82rem;
        color: var(--text-color);
        opacity: 0.75;
        line-height: 1.5;
    }
    .starters-heading {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text-color);
        margin: 1.4rem 0 0.6rem;
    }
    .starter-wrap {
        display: flex;
        flex-direction: column;
        min-height: 155px;
        border-radius: 12px;
        border: 1.5px solid;
        overflow: hidden;
        background: var(--secondary-background-color);
    }
    .starter-easy   { border-color: #22c55e; }
    .starter-medium { border-color: #f59e0b; }
    .starter-hard   { border-color: #ef4444; }
    .starter-body {
        flex: 1;
        padding: 0.85rem 1rem 0.75rem;
    }
    .starter-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .label-easy   { color: #22c55e; }
    .label-medium { color: #f59e0b; }
    .label-hard   { color: #ef4444; }
    .starter-text {
        font-size: 0.87rem;
        color: var(--text-color);
        font-weight: 500;
        line-height: 1.45;
    }"""

if OLD_CSS in content:
    content = content.replace(OLD_CSS, NEW_CSS)
    print("CSS replaced OK")
else:
    print("ERROR: CSS block not found in file")

# ── 2. Fix the corrupted welcome heading block with regex ─────────────────────
# The apostrophe in "Here's" is corrupted (\ufffd replacement character)
# Use a flexible regex that matches any character in its place
heading_pattern = re.compile(
    r"    st\.markdown\(\s*"
    r'"<p style=\'font-size:1\.05rem;font-weight:600;color:#1e293b;margin-bottom:0\.25rem;\'>"\s*'
    r'"What would you like to explore today\?"\s*'
    r'"</p>"\s*'
    r'"<p style=\'font-size:0\.85rem;color:#64748b;margin-bottom:1\.25rem;\'>"\s*'
    r'"[^"]+"\s*'
    r'"</p>",\s*'
    r"unsafe_allow_html=True,\s*"
    r"    \)",
    re.DOTALL
)

NEW_HEADING = (
    "    st.markdown(\n"
    '        "<div class=\'welcome-heading\'>What would you like to explore today?</div>"\n'
    '        "<div class=\'welcome-sub\'>This platform connects to four medical data sources \u2014 here is what each one contains:</div>",\n'
    "        unsafe_allow_html=True,\n"
    "    )"
)

new_content, n = heading_pattern.subn(NEW_HEADING, content)
if n:
    content = new_content
    print(f"Heading replaced OK ({n} occurrence)")
else:
    print("ERROR: heading pattern not matched")

# ── 3. Write back ─────────────────────────────────────────────────────────────
with open('d:/HybridRAG_Sanius/frontend/app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("File written successfully.")
