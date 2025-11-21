import streamlit as streamlit
import pandas as pd
import numpy as np
import re
from typing import Iterable, Tuple, Dict

# Create Email and Social Media Marketing Tracking EXIDs
PREFIXES = {
    'Email': 'gaa2',
    'SMS': 'haa2',
    'Instagram Link in Bio': 'bba2',
    'Instagram Story': 'dba2',
    'Facebook Post': 'daa2'
}

def _inc_code(code: str) -> str:
    # base-26 increment over A-Z for a 3-letter string (e.g., AAZ->ABA)
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise ValueError(f"Invalid code: {code}")
    letters = list(code)
    i = len(letters) - 1
    carry = 1
    while i >= 0 and carry:
        n = ord(letters[i]) - 65 + carry  # A=0
        carry, n = divmod(n, 26)
        letters[i] = chr(n + 65)
        i -= 1
    return "".join(letters)

def parse_exid(exid: str) -> Tuple[str, str]:
    m = re.fullmatch(r"([a-z0-9]+)_([A-Z]{3})", exid)
    if not m:
        raise ValueError(f"Bad EXID format: {exid}")
    return m.group(1), m.group(2)

def next_exid_for_channel(existing_exids: Iterable[str], channel_prefix: str) -> str:
    # filter by prefix, collect valid codes, and return prefix_nextCode
    codes = []
    seen = set()
    for ex in existing_exids:
        try:
            pfx, code = parse_exid(ex)
        except ValueError:
            continue  # ignore malformed
        if pfx == channel_prefix:
            codes.append(code)
            if ex in seen:
                # duplicate exact EXID
                pass
            seen.add(ex)
    if not codes:
        # Start wherever you like; using 'AAB' to match your historical pattern
        return f"{channel_prefix}_AAB"
    top = max(codes)  # lexicographic works with A–Z sequences
    return f"{channel_prefix}_{_inc_code(top)}"

def suggest_next_all(existing_exids: Iterable[str]) -> Dict[str, str]:
    return {
        "email": next_exid_for_channel(existing_exids, PREFIXES["Email"]),
        "Instagram Link in Bio": next_exid_for_channel(existing_exids, PREFIXES["Instagram Link in Bio"]),
        "Instagram Story": next_exid_for_channel(existing_exids, PREFIXES["Instagram Story"]),
        "Facebook Post": next_exid_for_channel(existing_exids, PREFIXES["Facebook Post"]),
        "SMS": next_exid_for_channel(existing_exids, PREFIXES["SMS"])
    }
username = streamlit.text_input("Username", value="", type='password')
password = streamlit.text_input("Password", value="", type='password')

if 1==1:#username == "admin" and password == "admin":
    #treamlit.success("Logged In!".format(username))
    ##streamlit.title("Build Your Marketing Tracking Link")
    #streamlit.write("Use the form below to create a custom marketing tracking link.")

    streamlit.write("### Step 1: Select Marketing Channel")
    col1, col2 = streamlit.columns(2)
    with col1:
        channels = ['Email', 'Organic Social', 'Paid Ads']
        channel_option = streamlit.selectbox("Step 1: Select Marketing Channel", channels)
    with col2:
        reserve_n = streamlit.number_input("Number of EXIDs to Reserve", min_value=1, max_value=5, value=1, step=1)

    if streamlit.button("Generate Next EXIDs"):
        prefix = PREFIXES.get(channel_option, None)
        suggestions = []
       # working = list(existing_exids)  # copy to mutate
        for _ in range(reserve_n):
            next_exid = next_exid_for_channel(suggestions, prefix)
            suggestions.append(next_exid)
            #working.append(next_exid)  # reserve it for next iteration
        streamlit.write("#### Suggested EXIDs:")
    streamlit.write(suggestions)

