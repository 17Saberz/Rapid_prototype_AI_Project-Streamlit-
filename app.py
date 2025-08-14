import streamlit as st
import numpy as np
import cv2
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Image Processing Demo", layout="wide")

st.title("🖼️ Streamlit Image Processing Playground")
st.caption("เลือกแหล่งภาพ → ปรับพารามิเตอร์ → ดูผลลัพธ์ + กราฟคุณสมบัติของภาพ")

# -----------------------------
# Sidebar: Source & Parameters
# -----------------------------
st.sidebar.header("⚙️ Processing Controls")

source = st.sidebar.selectbox(
    "เลือกแหล่งภาพ",
    ["Webcam (live)", "Webcam Snapshot", "Image URL", "Upload File"],
)

st.sidebar.subheader("🔧 ตัวเลือกการประมวลผล")
use_gray = st.sidebar.checkbox("แปลงเป็น Grayscale", value=False)

blur_ksize = st.sidebar.slider("Gaussian Blur kernel size (odd)", 1, 31, 7, step=2)
blur_sigma = st.sidebar.slider("Gaussian sigma", 0.0, 10.0, 1.2, step=0.1)
use_blur = st.sidebar.checkbox("ใช้ Gaussian Blur", value=True)

use_canny = st.sidebar.checkbox("ใช้ Canny Edge", value=True)
canny_t1 = st.sidebar.slider("Canny threshold1", 0, 255, 80, step=1)
canny_t2 = st.sidebar.slider("Canny threshold2", 0, 255, 160, step=1)

brightness = st.sidebar.slider("ปรับสว่าง (β)", -100, 100, 0, step=1)
contrast = st.sidebar.slider("ปรับคอนทราสต์ (α)", 0.1, 3.0, 1.0, step=0.1)

st.sidebar.info("Tip: ถ้าภาพเบลอมากไป ลองลด kernel size หรือ sigma")

# เก็บค่าพารามิเตอร์ใน session_state ให้ VideoProcessor ใช้ได้สด ๆ
st.session_state["proc_params"] = dict(
    use_gray=use_gray,
    use_blur=use_blur,
    blur_ksize=blur_ksize,
    blur_sigma=blur_sigma,
    use_canny=use_canny,
    canny_t1=canny_t1,
    canny_t2=canny_t2,
    brightness=brightness,
    contrast=contrast,
)

# -----------------------------
# Helper: processing pipeline
# -----------------------------
def apply_processing(bgr: np.ndarray, params: dict):
    img = bgr.copy()

    # ปรับสว่าง/คอนทราสต์: output = alpha*img + beta
    img = cv2.convertScaleAbs(img, alpha=params["contrast"], beta=params["brightness"])

    if params["use_gray"]:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        working = img
        is_gray = True
    else:
        working = img
        is_gray = False

    if params["use_blur"]:
        k = max(1, params["blur_ksize"])
        if k % 2 == 0:
            k += 1
        if is_gray:
            working = cv2.GaussianBlur(working, (k, k), params["blur_sigma"])
        else:
            working = cv2.GaussianBlur(working, (k, k), params["blur_sigma"])

    edges = None
    if params["use_canny"]:
        # หากยังเป็นสี ให้ทำเป็นเทาก่อนคำนวณขอบ
        canny_in = working if is_gray else cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(canny_in, params["canny_t1"], params["canny_t2"])

    # รวมผลลัพธ์แสดง
    if edges is not None:
        # ซ้อนเส้นขอบ (สีเขียว) ทับภาพที่ประมวลผล
        if is_gray:
            base_rgb = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
        else:
            base_rgb = working.copy()
        edge_rgb = np.zeros_like(base_rgb)
        edge_rgb[..., 1] = edges  # ใส่ใน channel G
        overlay = cv2.addWeighted(base_rgb, 1.0, edge_rgb, 0.8, 0)
        out_vis = overlay
    else:
        out_vis = working if not is_gray else cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)

    return out_vis, edges

def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# -----------------------------
# Helper: show plots
# -----------------------------
def show_hist_and_metrics(out_bgr: np.ndarray, edges: np.ndarray | None):
    col_h, col_m = st.columns([2, 1])

    with col_h:
        st.markdown("#### 📈 Histogram (ความเข้ม)")
        fig, ax = plt.subplots()
        if out_bgr.ndim == 3:
            rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            # แสดง histogram แยกช่องสี
            for i, ch in enumerate(["R", "G", "B"]):
                ax.hist(rgb[..., i].ravel(), bins=256, range=(0, 255), histtype='step', label=ch)
            ax.legend()
        else:
            ax.hist(out_bgr.ravel(), bins=256, range=(0, 255), histtype='step')
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    with col_h:
        st.markdown("#### 🔍 ค่าเฉลี่ย/ส่วนเบี่ยงเบนมาตรฐาน (Intensity)")
        if out_bgr.ndim == 3:
            gray = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = out_bgr
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        st.metric("Mean", f"{mean_val:.2f}")
        st.metric("Std", f"{std_val:.2f}")

    with col_m:
        st.markdown("#### ✨ Edge Density")
        if edges is not None:
            edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
            st.metric("สัดส่วนพิกเซลขอบ", f"{edge_ratio*100:.2f}%")
        else:
            st.write("—")

# -----------------------------
# Static image path (URL/Snapshot/Upload)
# -----------------------------
def run_static_pipeline(bgr: np.ndarray, title: str):
    st.subheader(title)
    out_vis, edges = apply_processing(bgr, st.session_state["proc_params"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ภาพต้นฉบับ**")
        st.image(to_rgb(bgr), channels="RGB", use_container_width=True)
    with c2:
        st.markdown("**ภาพหลังประมวลผล**")
        st.image(to_rgb(out_vis), channels="RGB", use_container_width=True)

    # ดาวน์โหลดผลลัพธ์
    out_rgb = to_rgb(out_vis)
    out_pil = Image.fromarray(out_rgb)
    buf = BytesIO()
    out_pil.save(buf, format="PNG")
    st.download_button("⬇️ ดาวน์โหลดผลลัพธ์ (PNG)", data=buf.getvalue(), file_name="processed.png", mime="image/png")

    show_hist_and_metrics(out_vis, edges)

# -----------------------------
# Webcam (live) via WebRTC
# -----------------------------
class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.params = st.session_state.get("proc_params", {})

    def recv(self, frame):
        # รับภาพ (RGB) → BGR
        img = frame.to_ndarray(format="bgr24")
        # อัปเดตพารามิเตอร์ล่าสุดทุกเฟรม
        self.params = st.session_state.get("proc_params", self.params)
        out_vis, _ = apply_processing(img, self.params)
        return out_vis

# TURN/STUN config (ใช้สาธารณะของ Google STUN)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -----------------------------
# Main: route by source
# -----------------------------
if source == "Webcam (live)":
    st.subheader("📷 Webcam (live)")
    st.write("อนุญาตกล้องจากเบราว์เซอร์ แล้วปรับพารามิเตอร์จาก Sidebar ได้ทันที")
    webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=LiveProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )
    st.info("หมายเหตุ: โหมด live ไม่แสดงกราฟ เนื่องจากเป็นสตรีมแบบเรียลไทม์ (เพื่อประสิทธิภาพ)")

elif source == "Webcam Snapshot":
    st.subheader("📸 Webcam Snapshot")
    snap = st.camera_input("ถ่ายภาพจากเว็บแคม")
    if snap is not None:
        image = Image.open(snap)
        bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        run_static_pipeline(bgr, "โหมด Snapshot")

elif source == "Image URL":
    st.subheader("🌐 Load จาก URL")
    url = st.text_input("วางลิงก์รูปภาพ (jpg/png)", value="https://images.unsplash.com/photo-1504208434309-cb69f4fe52b0")
    go = st.button("โหลดภาพ")
    if go and url:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            run_static_pipeline(bgr, "โหมด URL")
        except Exception as e:
            st.error(f"โหลดภาพไม่สำเร็จ: {e}")

elif source == "Upload File":
    st.subheader("📁 อัปโหลดไฟล์")
    file = st.file_uploader("อัปโหลดรูป (jpg/png)", type=["jpg", "jpeg", "png"])
    if file is not None:
        img = Image.open(file).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        run_static_pipeline(bgr, "โหมด Upload")

# ฟุตโน้ตเล็ก ๆ
st.markdown("---")
st.caption("ทำด้วย ❤️ ด้วย Streamlit + OpenCV | กราฟ: Histogram & Edge Density จากภาพหลังประมวลผล")
