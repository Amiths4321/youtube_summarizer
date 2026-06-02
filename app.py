import streamlit as st
import yt_dlp
import whisper
import requests
import json
import os
import tempfile
from datetime import datetime

# ---- Page Config ----
st.set_page_config(page_title="YouTube Summarizer", page_icon="📹", layout="wide")
st.title("📹 AI YouTube Video Summarizer")
st.write("Paste a YouTube link → AI transcribes and summarizes into key points.")

# ---- Sidebar: Ollama ----
st.sidebar.title("⚙️ Ollama Server")
server_ip   = st.sidebar.text_input("Remote Server IP", placeholder="e.g. 192.168.1.100")
server_port = st.sidebar.text_input("Port", value="11434")
model_name  = st.sidebar.selectbox("Model", ["llama3", "mistral", "phi3"])
ollama_url  = f"http://{server_ip}:{server_port}/api/generate"

# ---- Sidebar: Whisper Settings ----
st.sidebar.title("🎙️ Whisper Settings")
whisper_model = st.sidebar.selectbox(
    "Whisper Model",
    ["tiny", "base", "small", "medium", "large"],
    index=1,
    help="tiny=fastest, large=most accurate. Use 'base' to start."
)
whisper_language = st.sidebar.selectbox("Language", ["auto-detect", "en", "hi", "te", "ta", "mr"])

# ---- Main Input ----
st.subheader("🔗 YouTube URL")
youtube_url = st.text_input("Paste YouTube Link", placeholder="https://www.youtube.com/watch?v=...")

# ---- Output Options ----
st.subheader("📋 Summary Options")
col1, col2 = st.columns(2)
with col1:
    summary_style = st.selectbox("Summary Style", [
        "Key Points (Bullet List)",
        "Executive Summary (Short Paragraph)",
        "Detailed Notes with Timestamps",
        "Action Items & Takeaways",
        "ELI5 (Explain Like I'm 5)"
    ])
with col2:
    use_case = st.selectbox("I am watching this for", [
        "Learning / Education",
        "Business Research",
        "Sales / Marketing",
        "Real Estate / Property",
        "Technology / AI",
        "General Interest"
    ])

show_transcript = st.checkbox("Also show full transcript", value=False)

# ---- Run Button ----
if st.button("▶️ Download, Transcribe & Summarize"):

    if not youtube_url:
        st.warning("Please paste a YouTube URL.")
        st.stop()
    if not server_ip:
        st.error("Please enter your Ollama server IP in the sidebar.")
        st.stop()

    # ---- STEP A: Download Audio ----
    with st.spinner("📥 Downloading audio from YouTube..."):
        try:
            temp_dir    = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "audio.%(ext)s")

            ydl_opts = {
                "format":            "bestaudio/best",
                "outtmpl":           output_path,
                "postprocessors": [{
                    "key":            "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "128",
                }],
                "quiet": True
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info       = ydl.extract_info(youtube_url, download=True)
                video_title    = info.get("title", "Unknown Title")
                video_duration = info.get("duration", 0)
                video_channel  = info.get("uploader", "Unknown")

            audio_file = os.path.join(temp_dir, "audio.mp3")
            st.success(f"✅ Downloaded: **{video_title}**")

            col_i1, col_i2, col_i3 = st.columns(3)
            col_i1.metric("Channel", video_channel)
            col_i2.metric("Duration", f"{video_duration // 60} min {video_duration % 60} sec")
            col_i3.metric("Whisper Model", whisper_model)

        except Exception as e:
            st.error(f"❌ Failed to download: {str(e)}")
            st.stop()

    # ---- STEP B: Transcribe with Whisper ----
    with st.spinner(f"🎙️ Transcribing audio with Whisper ({whisper_model})... this may take a few minutes"):
        try:
            model = whisper.load_model(whisper_model)

            lang = None if whisper_language == "auto-detect" else whisper_language
            result = model.transcribe(audio_file, language=lang, fp16=False)

            transcript = result["text"].strip()
            st.success(f"✅ Transcription complete — {len(transcript.split())} words")

            if show_transcript:
                with st.expander("📄 Full Transcript"):
                    st.write(transcript)

            # Clean up audio file
            os.remove(audio_file)

        except Exception as e:
            st.error(f"❌ Transcription failed: {str(e)}")
            st.stop()

    # ---- STEP C: Summarize with Ollama ----
    with st.spinner(f"🧠 Summarizing with {model_name} on your GPU server..."):

        # Truncate transcript if very long (Ollama has context limits)
        max_words    = 3000
        words        = transcript.split()
        if len(words) > max_words:
            transcript_input = " ".join(words[:max_words]) + "\n\n[Transcript truncated for length...]"
            st.info(f"ℹ️ Transcript was long ({len(words)} words) — summarizing first {max_words} words.")
        else:
            transcript_input = transcript

        prompt = f"""
You are an expert content analyst.

Video Title: {video_title}
Channel: {video_channel}
Purpose of watching: {use_case}

Here is the video transcript:
{transcript_input}

Task: Summarize this video in the style: {summary_style}

Rules:
- Be concise and capture the most valuable insights
- Focus on what is relevant for: {use_case}
- If it is bullet points, use clear headings
- Highlight any specific numbers, facts, or actionable advice
- Write for someone who has NOT watched the video

Write the summary now:
"""

        try:
            response = requests.post(
                ollama_url,
                json={"model": model_name, "prompt": prompt, "stream": True},
                stream=True,
                timeout=180
            )

            summary    = ""
            output_box = st.empty()

            st.subheader("🧠 AI Summary")
            for line in response.iter_lines():
                if line:
                    data  = json.loads(line)
                    token = data.get("response", "")
                    summary += token
                    output_box.markdown(summary)
                    if data.get("done"):
                        break

            st.success("✅ Summary Complete!")

            # ---- Download Options ----
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            safe_title = video_title[:40].replace(" ", "_").replace("/", "-")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "📥 Download Summary (.txt)",
                    data=f"Video: {video_title}\nChannel: {video_channel}\nURL: {youtube_url}\nDate: {datetime.now().strftime('%d %b %Y')}\n\n{summary}",
                    file_name=f"summary_{safe_title}_{timestamp}.txt",
                    mime="text/plain"
                )
            with col_d2:
                if show_transcript:
                    st.download_button(
                        "📄 Download Full Transcript (.txt)",
                        data=transcript,
                        file_name=f"transcript_{safe_title}_{timestamp}.txt",
                        mime="text/plain"
                    )

        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to Ollama at {ollama_url}.")
        except Exception as e:
            st.error(f"Summarization error: {str(e)}")

# ---- Footer ----
st.markdown("---")
st.caption("yt-dlp + Whisper (local) + Ollama GPU | Zero API cost | 100% private")