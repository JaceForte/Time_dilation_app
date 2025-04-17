import streamlit as st
import re
import pandas as pd
import matplotlib.pyplot as plt
import openai
import io
from typing import List, Tuple

# --- Setup OpenAI API Key ---
from openai import OpenAI


# --- Regex Patterns ---
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

from openai import OpenAI  # Make sure this is still near the top

@st.cache_data
def call_gpt_api(transcript_chunk: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # ← Move client here

    prompt = f"""
You are analyzing an earnings call transcript. Extract all statements that refer to changes in timing (delays, accelerations, rescheduling, unexpected completions, etc). Return a markdown table with:
| Timeline Signal | Sentiment | Quote |

Transcript:
{transcript_chunk[:5000]}
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content

def parse_gpt_markdown_table(md: str) -> pd.DataFrame:
    lines = [line.strip() for line in md.splitlines() if '|' in line and not line.startswith('|---')]
    rows = [line.split('|')[1:-1] for line in lines if len(line.split('|')) >= 4]
    clean_rows = [tuple(cell.strip() for cell in row) for row in rows]
    return pd.DataFrame(clean_rows, columns=["Timeline Signal", "Sentiment", "Quote"])

def chunk_text(text: str, max_length: int = 4000) -> List[str]:
    paragraphs = text.split('\n')
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < max_length:
            current += para + "\n"
        else:
            chunks.append(current)
            current = para + "\n"
    if current:
        chunks.append(current)
    return chunks

def display_timeline_chart(df: pd.DataFrame):
    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df["Sentiment Score"] = df["Sentiment"].map(sentiment_map)
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["Sentiment Score"], marker='o')
    plt.yticks([-1, 0, 1], ["Negative", "Neutral", "Positive"])
    plt.title("Sentiment Over Time (GPT Events)")
    plt.xlabel("Event Order")
    plt.ylabel("Sentiment")
    st.pyplot(plt)

# --- Streamlit App ---
st.title("⏳ Time Dilation Event Tracker")
st.markdown("""
Upload an earnings call transcript or paste it below. This tool identifies statements related to **timeline shifts**, like delays or accelerations, using Regex and GPT.
""")

input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])
transcript = ""
uploaded_file = None

if input_method == "Paste Text":
    transcript = st.text_area("Paste transcript text here:", height=300)

elif input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload a .txt file with transcript text:")
    if uploaded_file is not None:
        try:
            transcript = uploaded_file.read().decode("utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            transcript = uploaded_file.read().decode("latin-1")

if transcript:
    st.subheader("🔍 Regex-Based Detection")
    regex_results = extract_time_signals(transcript)
    if regex_results:
        df_regex = pd.DataFrame(regex_results, columns=["Time Phrase", "Sentiment", "Context"])
        st.dataframe(df_regex)
        st.download_button("Download Regex Results", df_regex.to_csv(index=False), "regex_results.csv", "text/csv")
    else:
        st.info("No time-based phrases detected by regex.")

    if st.button("🧠 Use GPT to Analyze Timing Statements"):
        st.subheader("GPT Results")
        chunks = chunk_text(transcript)
        all_tables = []
        for i, chunk in enumerate(chunks):
            st.text(f"Processing chunk {i+1}/{len(chunks)}...")
            md_table = call_gpt_api(chunk)
            try:
                df_chunk = parse_gpt_markdown_table(md_table)
                all_tables.append(df_chunk)
            except Exception as e:
                st.warning(f"Failed to parse GPT output for chunk {i+1}: {e}")

        if all_tables:
            full_gpt_df = pd.concat(all_tables, ignore_index=True)
            st.dataframe(full_gpt_df)
            st.download_button("Download GPT Results", full_gpt_df.to_csv(index=False), "gpt_time_signals.csv", "text/csv")
            display_timeline_chart(full_gpt_df)
        else:
            st.info("GPT did not return any structured time signal statements.")
