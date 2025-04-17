import streamlit as st
import re
import pandas as pd
from typing import List, Tuple

# --- Helper Functions ---
TIME_PHRASES = [
    r"ahead of schedule",
    r"behind schedule",
    r"on track",
    r"delayed",
    r"postponed",
    r"accelerated",
    r"slower than expected",
    r"faster than expected",
    r"in the next quarter",
    r"in the coming months",
    r"pushed (back|forward)",
    r"moved (up|back)",
    r"reschedul(ed|ing)",
    r"timeline",
    r"timetable",
    r"expected to (complete|launch|begin|wrap up|ship)",
    r"by (mid|end|early)-?(Q[1-4]|year|month)",
    r"(next|this) (quarter|year|month)",
    r"mid-(year|quarter)",
    r"(longer|sooner|earlier|later) than (expected|planned|anticipated)",
    r"(completed|delivered) (earlier|later) than expected",
    r"as (scheduled|planned)",
    r"(project|launch|rollout|deployment).*?delayed"
]

SENTIMENT_MAP = {
    "ahead of schedule": "Positive",
    "accelerated": "Positive",
    "faster than expected": "Positive",
    "moved up": "Positive",
    "on track": "Neutral",
    "timeline": "Neutral",
    "timetable": "Neutral",
    "in the next quarter": "Neutral",
    "in the coming months": "Neutral",
    "by year-end": "Neutral",
    "this quarter": "Neutral",
    "next year": "Neutral",
    "mid-year": "Neutral",
    "behind schedule": "Negative",
    "delayed": "Negative",
    "postponed": "Negative",
    "slower than expected": "Negative",
    "pushed back": "Negative",
    "rescheduled": "Negative",
    "longer than expected": "Negative"
}

def extract_time_signals(text: str) -> List[Tuple[str, str, str]]:
    results = []
    for phrase in TIME_PHRASES:
        matches = re.finditer(phrase, text, flags=re.IGNORECASE)
        for match in matches:
            start = max(match.start() - 100, 0)
            end = min(match.end() + 100, len(text))
            quote = text[start:end].replace("\n", " ").strip()
            sentiment = SENTIMENT_MAP.get(phrase.lower(), "Unknown")
            results.append((phrase, sentiment, quote))
    return results

# --- Streamlit App ---
st.title("⏳ Time Dilation Event Tracker")
st.markdown("""
Upload an earnings call transcript or paste it below. This tool identifies statements related to **timeline shifts**, like delays or accelerations, and classifies their sentiment.
""")

input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])
transcript = ""
uploaded_file = None  # Initialize to avoid NameError

if input_method == "Paste Text":
    transcript = st.text_area("Paste transcript text here:", height=300)

elif input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload a .txt file with transcript text:")
    if uploaded_file is not None:
        try:
            transcript = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)  # Reset file pointer
            transcript = uploaded_file.read().decode("latin-1")

if transcript:
    st.write("\n## Detected Time-Based Statements")
    results = extract_time_signals(transcript)

    if results:
        df = pd.DataFrame(results, columns=["Time Phrase", "Sentiment", "Context"])
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "time_dilation_events.csv", "text/csv")
    else:
        st.info("No time-based phrases detected.")
