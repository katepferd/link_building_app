import re
import io
import urllib.parse as up
from typing import Iterable, Tuple, Dict, List

import pandas as pd
import streamlit as st

# ------------------------------
# Core EXID logic
# ------------------------------
PREFIXES = {
    'Email': 'gaa2',
    'SMS': 'haa2',
    'Instagram Link in Bio': 'bba2',
    'Instagram Story': 'dba2',
    'Facebook Post': 'daa2'
}

EXID_RE = re.compile(r"^([a-z0-9]+)_([A-Z]{3})$")


def inc_code(code: str) -> str:
    """Base-26 increment over A–Z for a 3-letter code (e.g., AAZ -> ABA)."""
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise ValueError(f"Invalid code: {code}")
    letters = list(code)
    i, carry = 2, 1
    while i >= 0 and carry:
        n = (ord(letters[i]) - 65) + carry
        carry, n = divmod(n, 26)
        letters[i] = chr(n + 65)
        i -= 1
    return "".join(letters)


def parse_exid(exid: str) -> Tuple[str, str]:
    m = EXID_RE.fullmatch(exid)
    if not m:
        raise ValueError(f"Bad EXID format: {exid}")
    return m.group(1), m.group(2)


def next_exid_for_channel(existing_exids: Iterable[str], channel_prefix: str, default_start: str = "AAA") -> str:
    """Return next EXID for a given prefix based on max observed 3-letter code."""
    codes: List[str] = []
    for ex in existing_exids:
        try:
            pfx, code = parse_exid(ex)
        except ValueError:
            continue
        if pfx == channel_prefix:
            # Validate only A–Z letters; skip malformed like AB0
            if re.fullmatch(r"[A-Z]{3}", code):
                codes.append(code)
    if not codes:
        return f"{channel_prefix}_{default_start}"
    top = max(codes)  # Lexicographic OK for A–Z triplets
    return f"{channel_prefix}_{inc_code(top)}"


def next_n_for_channel(existing_exids: Iterable[str], channel_prefix: str, n: int) -> List[str]:
    """Reserve a consecutive block of n EXIDs for a channel."""
    results = []
    # Create a working copy so increments cascade
    working = list(existing_exids)
    for _ in range(n):
        nxt = next_exid_for_channel(working, channel_prefix)
        results.append(nxt)
        working.append(nxt)
    return results


# ------------------------------
# URL utilities
# ------------------------------

def apply_exid_to_url(url: str, exid: str) -> Tuple[str, List[str]]:
    """Return URL with EXID param applied. Collect warnings as list of strings.
    - If an EXID param already exists and differs, it is replaced and a warning is emitted.
    - If URL is suspicious (not http/https), warn.
    """
    warns = []
    url = url.strip()
    if not url:
        return url, warns

    if not (url.startswith("http://") or url.startswith("https://")):
        warns.append("URL does not start with http/https — left unchanged")
        return url, warns

    try:
        parsed = up.urlparse(url)
        q = up.parse_qs(parsed.query, keep_blank_values=True)

        # Normalize incoming EXID parameter(s)
        existing_vals = q.get("EXID", [])
        if existing_vals and existing_vals[0] != exid:
            warns.append(f"Existing EXID={existing_vals[0]} replaced with {exid}")
        q["EXID"] = [exid]

        new_query = up.urlencode(q, doseq=True)
        new_url = up.urlunparse(parsed._replace(query=new_query))
        return new_url, warns
    except Exception as e:
        warns.append(f"Failed to parse URL ({e}) — left unchanged")
        return url, warns


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="EXID Builder • Young Living", layout="wide")
st.title("EXID Builder")

st.markdown(
    """
This app helps you **generate, reserve, and apply** EXIDs to links for Email (`gaa2`) and SMS (`haa2`).

**Workflow**
1) Load your existing EXID registry (paste or upload CSV).  
2) Inspect the latest per channel.  
3) Generate the next EXID(s) for your campaign.  
4) Paste campaign links — we will append/replace the `EXID` query param.  
5) Export a CSV and/or append new rows to your in-session registry.

> Tip: keep a single team CSV as the source of truth to prevent collisions.
"""
)

"""with st.expander("1) Load existing EXIDs", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        exids_text = st.text_area(
            "Paste existing EXIDs (one per line)",
            height=240,
            placeholder="e.g.\ngaa2_AAB\ngaa2_AAC\nhaa2_AAA"
        )
        pasted = [x.strip() for x in exids_text.splitlines() if x.strip()]

    with c2:
        uploaded = st.file_uploader("...or upload a CSV with at least an `EXID` column", type=["csv"]) 
        from_csv: List[str] = []
        if uploaded:
            try:
                dfu = pd.read_csv(uploaded)
                if "EXID" in dfu.columns:
                    from_csv = dfu["EXID"].dropna().astype(str).tolist()
                else:
                    st.warning("No EXID column found; ignoring file.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    existing_exids = pasted + from_csv

    # Show quick stats / validation
    if existing_exids:
        df_show = pd.DataFrame({"EXID": existing_exids})
        df_show["Valid Format"] = df_show["EXID"].apply(lambda x: bool(EXID_RE.fullmatch(x)))
        dupes = df_show["EXID"].duplicated(keep=False)
        df_show["Duplicate"] = dupes
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Latest per prefix
        latest = {}
        for label, prefix in PREFIXES.items():
            codes = []
            for ex in existing_exids:
                try:
                    pfx, code = parse_exid(ex)
                except ValueError:
                    continue
                if pfx == prefix and re.fullmatch(r"[A-Z]{3}", code):
                    codes.append(code)
            if codes:
                latest[label] = f"{prefix}_{max(codes)}"
            else:
                latest[label] = "(none)"
        st.info("Latest seen → " + ", ".join([f"{k}: {v}" for k, v in latest.items()]))
    else:
        st.warning("No EXIDs loaded yet. We'll start from AAB for each channel by default.")
"""
st.divider()

with st.expander("2) Generate EXID(s) for a campaign", expanded=True):
    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        campaign_name = st.text_input("Campaign name", placeholder="e.g., Black Friday Launch")
    with colB:
        channel_label = st.selectbox("Channel", list(PREFIXES.keys()))
        channel_prefix = PREFIXES[channel_label]
    with colC:
        reserve_n = st.number_input("How many to reserve?", min_value=1, max_value=20, value=1, step=1,
                                    help="Use >1 for A/B tests or multi-link variants.")

    manual_override = st.text_input(
        "Optional: override next starting code (3 letters, A–Z)",
        help="If set (e.g., ADL), allocation will begin with that code regardless of history.")

    urls_text = st.text_area(
        "Paste one or more URLs (one per line) to tag with the EXID(s)", height=200,
        placeholder="https://www.youngliving.com/us/en/category/best-sellers\nhttps://www.youngliving.com/us/en/product/vanilla-mint-essential-oil-blend"
    )

    go = st.button("Generate EXIDs and tagged links", type="primary")

    if go:
        # Compute the block of EXIDs
        start_block: List[str]
        if manual_override:
            start_code = manual_override.strip().upper()
            if not re.fullmatch(r"[A-Z]{3}", start_code):
                st.error("Manual override must be exactly 3 letters A–Z (e.g., ADL).")
                st.stop()
            first_exid = f"{channel_prefix}_{start_code}"
            # Build N codes ascending from manual start
            block = [first_exid]
            code = start_code
            for _ in range(reserve_n - 1):
                code = inc_code(code)
                block.append(f"{channel_prefix}_{code}")
            start_block = block
        else:
            start_block = next_n_for_channel(existing_exids, channel_prefix, reserve_n)

        # Prepare URLs
        raw_urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if raw_urls and len(raw_urls) not in (1, len(start_block)):
            st.warning(
                "Number of URLs doesn't match the number of EXIDs. We'll apply the **first EXID to all URLs**.\n"
                "If you want one EXID per URL, supply the same count.")

        # Expand rows
        rows = []
        warns_global: List[str] = []
        for i, exid in enumerate(start_block):
            assignment_urls: List[str]
            if not raw_urls:
                assignment_urls = [""]
            elif len(raw_urls) == len(start_block):
                assignment_urls = [raw_urls[i]]
            else:
                assignment_urls = raw_urls  # apply first EXID to all URLs

            variant = chr(65 + i) if reserve_n > 1 else "A"

            for url in assignment_urls:
                final_url, warns = apply_exid_to_url(url, exid)
                rows.append({
                    "Channel": channel_label,
                    "Campaign": campaign_name or "(untitled)",
                    "EXID": exid,
                    "Tagged URL": final_url,
                })
                warns_global.extend([f"{exid}: {w}" for w in warns])

        out_df = pd.DataFrame(rows)
        st.success(f"Generated {len(start_block)} EXID(s): {', '.join(start_block)}")
        if warns_global:
            st.warning("\n".join(sorted(set(warns_global))))

        st.dataframe(out_df, use_container_width=True, hide_index=True)

        # Download
        csv = out_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="exid_assignments.csv",
            mime="text/csv",
        )

        # Append to session registry
        if "registry" not in st.session_state:
            st.session_state.registry = []
        append_clicked = st.checkbox("Append these EXIDs to my in-session registry (prevents reuse in this session)")
        if append_clicked:
            st.session_state.registry.extend([r["EXID"] for r in rows])
            st.info(f"Session registry now contains {len(st.session_state.registry)} EXID rows (may include duplicates for multi-link rows).")

st.divider()

with st.expander("3) Session registry (optional)", expanded=False):
    st.markdown("The session registry accumulates EXIDs you append during this session.")
    reg = st.session_state.get("registry", [])
    if reg:
        df_reg = pd.DataFrame({"EXID": reg})
        st.dataframe(df_reg, use_container_width=True, hide_index=True)
        st.download_button(
            "Download session registry (CSV)",
            data=df_reg.to_csv(index=False).encode("utf-8"),
            file_name="exid_session_registry.csv",
            mime="text/csv",
        )
    else:
        st.write("(empty)")

st.divider()

with st.expander("Notes & Guardrails", expanded=False):
    st.markdown(
        """
- **Validation**: malformed codes (e.g., `AB0` with a zero) are ignored when computing the next value.
- **Conflicts**: if a pasted URL already has an `EXID`, the app will **replace** it and show a warning.
- **Manual start**: use the override to begin at an exact code for backfills or re-aligning sequences.
- **Best practice**: maintain a single team CSV with columns like `EXID, Campaign, Channel, Owner, Date, URLs...` and import it here before assigning.
        """
    )

