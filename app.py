
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from collections import defaultdict
from ultralytics import YOLO
import plotly.express as px
import pandas as pd

# -----------------------------------------
# SENTRAPULSE - AI CROWD MONITORING SYSTEM
# -----------------------------------------

st.set_page_config(
    page_title="SentraPulse | Crowd Intelligence",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.metric-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ SentraPulse")
st.subheader("AI-Powered Smart Crowd Monitoring System")

st.caption(
    "Predict Risks Early. Protect People Faster."
)

st.info(
    "Upload CCTV footage to detect people, track "
    "movement, estimate crowd density and identify "
    "potential congestion."
)

# -----------------------------------------
# SIDEBAR SETTINGS
# -----------------------------------------

st.sidebar.header("⚙️ System Settings")

confidence = st.sidebar.slider(
    "Detection Confidence",
    0.10, 0.90, 0.40, 0.05
)

yellow_limit = st.sidebar.number_input(
    "Caution Count",
    min_value=1,
    value=15
)

orange_limit = st.sidebar.number_input(
    "Warning Count",
    min_value=2,
    value=25
)

red_limit = st.sidebar.number_input(
    "High Risk Count",
    min_value=3,
    value=40
)

model_name = st.sidebar.selectbox(
    "AI Detection Model",
    ["yolo11n.pt", "yolov8n.pt"]
)

show_heatmap = st.sidebar.checkbox(
    "Show Crowd Heatmap", True
)

# -----------------------------------------
# LOAD YOLO MODEL
# -----------------------------------------

@st.cache_resource
def load_model(name):
    return YOLO(name)

# -----------------------------------------
# RISK ANALYSIS
# -----------------------------------------

def get_risk(count, slow_count):

    if count >= red_limit:
        return "RED - HIGH RISK", "🔴"

    if count >= orange_limit and slow_count >= 3:
        return "ORANGE - WARNING", "🟠"

    if count >= yellow_limit:
        return "YELLOW - CAUTION", "🟡"

    return "GREEN - NORMAL", "🟢"


# -----------------------------------------
# MOVEMENT DIRECTION
# -----------------------------------------

def get_direction(dx, dy):

    if abs(dx) < 2 and abs(dy) < 2:
        return "Stationary"

    angle = np.degrees(np.arctan2(-dy, dx))

    if -22.5 <= angle < 22.5:
        return "Right"

    elif 22.5 <= angle < 67.5:
        return "Up-Right"

    elif 67.5 <= angle < 112.5:
        return "Up"

    elif 112.5 <= angle < 157.5:
        return "Up-Left"

    elif angle >= 157.5 or angle < -157.5:
        return "Left"

    elif -157.5 <= angle < -112.5:
        return "Down-Left"

    elif -112.5 <= angle < -67.5:
        return "Down"

    return "Down-Right"


# -----------------------------------------
# VIDEO UPLOAD
# -----------------------------------------

uploaded_video = st.file_uploader(
    "📹 Upload CCTV / Crowd Video",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button("🚀 Analyze Crowd Video", type="primary"):

        model = load_model(model_name)

        # Save uploaded video temporarily
        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(uploaded_video.getvalue())
        input_file.close()

        cap = cv2.VideoCapture(input_file.name)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            st.error("Unable to read this video.")
            cap.release()
            os.unlink(input_file.name)
            st.stop()

        # Output video
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_file.close()

        writer = cv2.VideoWriter(
            output_file.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        # Tracking information
        unique_people = set()
        last_positions = {}
        speed_values = defaultdict(list)

        # Heatmap
        heatmap = np.zeros(
            (height, width),
            dtype=np.float32
        )

        # Statistics
        frame_counts = []
        frame_numbers = []
        total_frames = 0
        peak_count = 0
        slow_count = 0

        progress = st.progress(0)
        status = st.empty()

        max_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # ---------------------------------
        # PROCESS VIDEO FRAME BY FRAME
        # ---------------------------------

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            total_frames += 1

            # AI detection and tracking
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=confidence,
                verbose=False
            )

            result = results[0]

            current_count = 0
            slow_count = 0
            current_speeds = []

            if result.boxes is not None:

                boxes = result.boxes

                if boxes.id is not None:

                    ids = boxes.id.cpu().numpy().astype(int)
                    coordinates = boxes.xyxy.cpu().numpy()

                    for box, person_id in zip(
                        coordinates, ids
                    ):

                        person_id = int(person_id)

                        x1, y1, x2, y2 = map(
                            int, box
                        )

                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        current_count += 1
                        unique_people.add(person_id)

                        # Draw person bounding box
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        # Track previous position
                        speed = 0.0
                        direction = "New"

                        if person_id in last_positions:

                            px, py, previous_frame = (
                                last_positions[person_id]
                            )

                            dx = cx - px
                            dy = cy - py

                            elapsed = (
                                total_frames - previous_frame
                            ) / fps

                            if elapsed > 0:

                                speed = (
                                    np.sqrt(dx**2 + dy**2)
                                    / elapsed
                                )

                            direction = get_direction(
                                dx, dy
                            )

                            # Draw movement arrow
                            if np.sqrt(dx**2 + dy**2) > 3:

                                cv2.arrowedLine(
                                    frame,
                                    (px, py),
                                    (cx, cy),
                                    (255, 0, 0),
                                    2,
                                    tipLength=0.3
                                )

                        last_positions[person_id] = (
                            cx, cy, total_frames
                        )

                        speed_values[person_id].append(
                            speed
                        )

                        current_speeds.append(speed)

                        if speed < 15:
                            slow_count += 1

                        # Add to crowd heatmap
                        cv2.circle(
                            heatmap,
                            (cx, cy),
                            20,
                            1,
                            -1
                        )

                        # Person ID and direction
                        label = (
                            f"ID {person_id} | {direction}"
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (0, 255, 255),
                            1
                        )

            # Risk analysis
            risk, risk_icon = get_risk(
                current_count,
                slow_count
            )

            peak_count = max(
                peak_count,
                current_count
            )

            frame_counts.append(current_count)
            frame_numbers.append(total_frames)

            # Video information overlay
            cv2.rectangle(
                frame,
                (0, 0),
                (width, 95),
                (25, 25, 25),
                -1
            )

            cv2.putText(
                frame,
                f"PEOPLE: {current_count}",
                (15, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"UNIQUE: {len(unique_people)}",
                (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"RISK: {risk}",
                (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            writer.write(frame)

            if max_frames > 0:
                progress.progress(
                    min(total_frames / max_frames, 1.0)
                )

            status.text(
                f"Processing frame {total_frames} | "
                f"People detected: {current_count}"
            )

        cap.release()
        writer.release()

        os.unlink(input_file.name)

        progress.progress(1.0)
        status.success("Video analysis completed!")

        # ---------------------------------
        # FINAL STATISTICS
        # ---------------------------------

        final_count = (
            frame_counts[-1] if frame_counts else 0
        )

        all_speeds = [
            speed
            for values in speed_values.values()
            for speed in values
            if speed > 0
        ]

        average_speed = (
            np.mean(all_speeds)
            if all_speeds else 0
        )

        final_slow_count = (
            sum(
                1 for values in speed_values.values()
                if values and values[-1] < 15
            )
        )

        final_risk, final_icon = get_risk(
            final_count,
            final_slow_count
        )

        # ---------------------------------
        # DASHBOARD
        # ---------------------------------

        st.divider()

        st.header("📊 SentraPulse Analytics Dashboard")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Unique People",
            len(unique_people)
        )

        c2.metric(
            "Current People",
            final_count
        )

        c3.metric(
            "Peak People",
            peak_count
        )

        c4.metric(
            "Average Speed",
            f"{average_speed:.1f} px/s"
        )

        st.subheader("🚨 Crowd Risk Status")

        st.warning(
            f"{final_icon} {final_risk}"
        )

        st.caption(
            "Risk status is based on configurable prototype "
            "thresholds, not a validated emergency prediction."
        )

        # ---------------------------------
        # PEOPLE COUNT GRAPH
        # ---------------------------------

        st.subheader("People Count Over Time")

        graph_data = pd.DataFrame({
            "Frame": frame_numbers,
            "People": frame_counts
        })

        fig = px.line(
            graph_data,
            x="Frame",
            y="People",
            title="Crowd Count Analysis"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ---------------------------------
        # MOVEMENT ANALYSIS
        # ---------------------------------

        st.subheader("🏃 Movement Analysis")

        direction_counts = defaultdict(int)

        for person_id, values in speed_values.items():

            if len(values) > 1:
                direction_counts[person_id] += 1

        st.metric(
            "People Tracked",
            len(unique_people)
        )

        st.metric(
            "Tracked Movement Records",
            sum(direction_counts.values())
        )

        st.write(
            "Average estimated movement speed:",
            f"{average_speed:.2f} pixels/second"
        )

        st.caption(
            "Speed is estimated in image pixels. "
            "Real-world speed in metres/second requires "
            "camera calibration."
        )

        # ---------------------------------
        # HEATMAP
        # ---------------------------------

        if show_heatmap:

            st.subheader("🔥 Crowd Movement Heatmap")

            normalized = cv2.normalize(
                heatmap,
                None,
                0,
                255,
                cv2.NORM_MINMAX
            ).astype(np.uint8)

            colored_heatmap = cv2.applyColorMap(
                normalized,
                cv2.COLORMAP_JET
            )

            heatmap_rgb = cv2.cvtColor(
                colored_heatmap,
                cv2.COLOR_BGR2RGB
            )

            st.image(
                heatmap_rgb,
                caption="Crowd Position Density Heatmap",
                use_container_width=True
            )

        # ---------------------------------
        # PROCESSED VIDEO
        # ---------------------------------

        st.divider()

        st.header("🎥 Processed Crowd Video")

        if os.path.exists(output_file.name):

            with open(output_file.name, "rb") as video:
                video_bytes = video.read()

            st.video(video_bytes)

            st.download_button(
                label="⬇️ Download Processed Video",
                data=video_bytes,
                file_name="SentraPulse_Analysis.mp4",
                mime="video/mp4"
            )

            os.unlink(output_file.name)

        st.success(
            "SentraPulse analysis completed successfully!"
        )

else:

    st.info(
        "Please upload a CCTV video to start "
        "SentraPulse crowd analysis."
    )

st.divider()

st.caption(
    "SentraPulse | AI-Based Crowd Density Risk "
    "Prediction and Safety Alert System"
)

st.caption(
    "Prototype version | For experimental and "
    "demonstration purposes."
          )
          
