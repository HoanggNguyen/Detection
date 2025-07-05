import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import os
from PIL import Image
import pickle
import time
import tempfile

# --- PAGE CONFIGURATION & STYLING ---

st.set_page_config(
    page_title="Face Recognition App",
    page_icon="⚫",
    layout="wide"
)

def apply_custom_styling():
    """Injects custom CSS inspired by the provided design template."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

            /* --- General --- */
            html, body, [class*="st-"] {
                font-family: 'Space Grotesk', sans-serif;
                color: #191A23;
            }
            .main {
                background-color: #F9F9F9; /* Light grey bg for the page */
            }
            .main .block-container {
                padding: 2rem 2.5rem;
                background-color: white;
                border-radius: 20px;
                box-shadow: 0 8px 24px rgba(0,0,0,0.05);
            }

            /* --- Headings --- */
            h1 {
                font-weight: 700;
                padding-bottom: 0.5rem;
                border-bottom: none;
            }
            h2 {
                font-weight: 700;
                border-bottom: 2px solid #F0F2F6;
                padding-bottom: .75rem;
                margin-top: 1.5rem;
                margin-bottom: 1rem;
            }
            h3 { font-weight: 500; }

            /* --- Buttons --- */
            .stButton > button {
                border-radius: 12px;
                background-color: #191A23;
                color: white;
                border: 2px solid #191A23;
                padding: 10px 24px;
                font-weight: 500;
                transition: all 0.2s ease-in-out;
            }
            .stButton > button:hover {
                background-color: white;
                color: #191A23;
            }
            .stButton > button[kind="secondary"] {
                background-color: transparent;
                color: #FF4B4B;
                border: 2px solid #FF4B4B;
            }
            .stButton > button[kind="secondary"]:hover {
                background-color: #FF4B4B;
                color: white;
            }
            /* Special button for camera inside dark card */
            .dark-card .stButton > button {
                background-color: #B9FF66;
                color: #191A23;
                border: 2px solid #B9FF66;
            }
            .dark-card .stButton > button:hover {
                background-color: #191A23;
                color: #B9FF66;
            }

            /* --- Tabs --- */
            .stTabs [data-baseweb="tab-list"] {
        		gap: 16px;
                padding-bottom: 1px;
                border-bottom: 2px solid #F0F2F6;
        	}
        	.stTabs [data-baseweb="tab"] {
        		background-color: transparent;
        		border-radius: 8px 8px 0 0;
        		padding: 10px 20px;
                font-weight: 500;
                color: #6c757d;
                transition: all 0.2s;
        	}
        	.stTabs [aria-selected="true"] {
                color: #191A23;
                background-color: #B9FF66; /* Accent green for active tab */
        	}

            /* --- Sidebar --- */
            [data-testid="stSidebar"] {
                background-color: #FFFFFF;
                border-right: 1px solid #EAEAEA;
            }
            [data-testid="stSidebar"] h2 {
                border: none;
            }
        </style>
    """, unsafe_allow_html=True)


# --- SESSION STATE INITIALIZATION ---
if 'face_db' not in st.session_state:
    st.session_state.face_db = []
if 'app' not in st.session_state:
    st.session_state.app = None
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

# --- CORE FUNCTIONS (Unchanged) ---
@st.cache_resource
def load_face_analysis():
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=-1)
    return app

def save_face_db(face_db, filename="face_db.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(face_db, f)

def load_face_db(filename="face_db.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return []

def add_face_to_db(name, image, app):
    img_array = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    faces = app.get(img_bgr)
    if not faces:
        return False, "Không tìm thấy khuôn mặt trong ảnh."
    embedding = faces[0].embedding / norm(faces[0].embedding)
    st.session_state.face_db.append((name, embedding))
    save_face_db(st.session_state.face_db)
    return True, f"Đã thêm {name} vào cơ sở dữ liệu thành công!"

def find_best_match(embedding, db, threshold=0.4):
    embedding = embedding / norm(embedding)
    best_score = -1
    best_name = "Unknown"
    for name, db_emb in db:
        sim = np.dot(embedding, db_emb)
        if sim > best_score and sim > threshold:
            best_score = sim
            best_name = f"{name} ({sim:.2f})"
    return best_name

def recognize_faces_in_image(image, app, face_db):
    img_array = np.array(image.convert('RGB'))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    faces = app.get(img_bgr)
    for face in faces:
        box = face.bbox.astype(int)
        name = find_best_match(face.embedding, face_db)
        cv2.rectangle(img_bgr, (box[0], box[1]), (box[2], box[3]), (185, 255, 102), 2)
        (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_bgr, (box[0], box[1] - h - 15), (box[0] + w, box[1] - 5), (185, 255, 102), -1)
        cv2.putText(img_bgr, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

# --- CAMERA & VIDEO PROCESSING FUNCTIONS ---
def init_camera():
    if st.session_state.cap is None:
        cap = cv2.VideoCapture(1)
        if not cap.isOpened(): cap = cv2.VideoCapture(0)
        if cap.isOpened():
            st.session_state.cap = cap
            return True
        else:
            st.error("Không thể mở camera. Vui lòng kiểm tra kết nối thiết bị.")
            st.session_state.camera_active = False
            return False
    return True

def release_camera():
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

def process_frame(frame, app, face_db):
    if frame is None: return None
    faces = app.get(frame)
    for face in faces:
        box = face.bbox.astype(int)
        name = find_best_match(face.embedding, face_db)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (185, 255, 102), 2)
        (w, h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (box[0], box[1] - h - 15), (box[0] + w, box[1] - 5), (185, 255, 102), -1)
        cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return frame

# --- MAIN APPLICATION UI ---
def main():
    apply_custom_styling()

    # --- MODEL & DB LOADING ---
    if st.session_state.app is None:
        with st.spinner("Đang tải model nhận diện..."):
            st.session_state.app = load_face_analysis()
    if not st.session_state.face_db:
        st.session_state.face_db = load_face_db()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚫ Database Info")
        st.write(f"Số lượng người đã đăng ký: **{len(st.session_state.face_db)}**")
        if st.session_state.face_db:
            st.write("Danh sách:")
            for name, _ in st.session_state.face_db:
                st.write(f"• {name}")
        else:
            st.info("Database đang trống.")

    # --- MAIN PAGE ---
    st.markdown("""
        <h1 style="font-size: 3rem; text-align: center; padding: 1.5rem; border-radius: 1rem;
                   background-color: #fff0f0; color: #FF4B4B; line-height: 1.4;">
            Nhận Diện Khuôn Mặt - ArcFace
        </h1>
    """, unsafe_allow_html=True)
    st.header("Mô tả")
    st.markdown("""
    Phần demo nhận diện khuôn mặt của nhóm sử dụng **độ lỗi ArcFace**, được thực hiện bằng cách gọi mô hình `buffalo_l` — một mô hình đã được huấn luyện bằng hàm **ArcFace Loss**.
    - **Detection Model**: RetinaFace-10GF 
    - **Recognition Model**: ResNet50@WebFace600K 
    - **Model Size**: 326MB  
    """)

    # --- TABS ---
    tab_image, tab_video, tab_live, tab_add, tab_manage = st.tabs(["Ảnh", "Video", "Thời gian thực", "Thêm khuôn mặt", "Quản lý Database"])

    with tab_image:
        st.header("Nhận diện khuôn mặt từ ảnh")
        uploaded_file = st.file_uploader("Tải lên một ảnh", type=['jpg', 'jpeg', 'png'], key="uploader_img")
        if uploaded_file:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Ảnh Gốc")
                st.image(image, use_column_width=True, channels="RGB")
            with col2:
                st.subheader("Kết Quả Nhận Diện")
                with st.spinner("Đang xử lý..."):
                    result_image = recognize_faces_in_image(image, st.session_state.app, st.session_state.face_db)
                    st.image(result_image, use_column_width=True)

    with tab_video:
        st.header("Nhận diện khuôn mặt từ video")
        uploaded_video = st.file_uploader("Tải lên một video", type=['mp4', 'mov', 'avi'], key="uploader_vid")
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Video Gốc")
                original_video_placeholder = st.empty()
            with col2:
                st.subheader("Video Đã Xử Lý")
                processed_video_placeholder = st.empty()

            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Lỗi: Không thể mở file video.")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        original_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        original_video_placeholder.image(original_frame_rgb, channels="RGB", use_column_width=True)
                        
                        processed_frame = process_frame(frame, st.session_state.app, st.session_state.face_db)
                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        processed_video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                        
                        frame_count += 1
                        progress_percentage = int((frame_count / total_frames) * 100)
                        progress_bar.progress(progress_percentage)
                        status_text.text(f"Đang xử lý frame: {frame_count}/{total_frames}")

                    status_text.success("Xử lý video hoàn tất!")
            finally:
                if 'cap' in locals() and cap.isOpened():
                    cap.release()
                os.remove(video_path)

    with tab_live:
        st.header("Nhận diện khuôn mặt thời gian thực")

        def toggle_camera():
            st.session_state.camera_active = not st.session_state.camera_active
            if not st.session_state.camera_active:
                release_camera()

        st.button(
            'Bật Camera' if not st.session_state.camera_active else 'Dừng Camera',
            on_click=toggle_camera,
            key="camera_toggle"
        )

        _, col_cam_display, _ = st.columns([1, 4, 1])
        with col_cam_display:
            camera_placeholder = st.empty()
            if st.session_state.camera_active:
                if init_camera():
                    while st.session_state.camera_active and st.session_state.cap.isOpened():
                        ret, frame = st.session_state.cap.read()
                        if not ret:
                            st.error("Không thể đọc frame từ camera.")
                            st.session_state.camera_active = False
                            release_camera()
                            st.rerun()
                            break
                        
                        processed_frame = process_frame(frame, st.session_state.app, st.session_state.face_db)
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, caption="Luồng camera trực tiếp", use_column_width=True)
                else:
                    camera_placeholder.error("Không thể khởi tạo camera.")
            else:
                release_camera()

    with tab_add:
        st.header("Thêm khuôn mặt mới vào Database")
        col1, col2 = st.columns([1, 1])
        with col1:
            name = st.text_input("Tên người:", placeholder="Nhập tên tại đây...")
            uploaded_file_add = st.file_uploader("Tải lên ảnh khuôn mặt (rõ nét):", type=['jpg', 'jpeg', 'png'], key="add_face_uploader")
            if name.strip() and uploaded_file_add:
                if st.button("➕ Thêm vào Database"):
                    image = Image.open(uploaded_file_add)
                    with st.spinner("Đang phân tích và lưu trữ..."):
                        success, message = add_face_to_db(name.strip(), image, st.session_state.app)
                        if success: st.success(message)
                        else: st.error(message)
        with col2:
            if uploaded_file_add:
                st.image(uploaded_file_add, caption="Xem trước ảnh tải lên", use_column_width=True)
            else:
                st.info("Vui lòng tải lên một ảnh để xem trước.")

    with tab_manage:
        st.header("Quản lý Database")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Xóa toàn bộ Database", type="secondary"):
                if st.session_state.face_db:
                    st.session_state.face_db = []
                    save_face_db([])
                    st.success("Đã xóa toàn bộ database!")
                else: st.info("Database đã trống!")
        with col2:
            if st.button("🔄 Tải lại Database từ File"):
                st.session_state.face_db = load_face_db()
                st.success("Đã tải lại database thành công!")
        
        st.subheader("Thông tin hệ thống")
        status_icon = '🟢' if st.session_state.camera_active else '🔴'
        status_text = 'Đang hoạt động' if st.session_state.camera_active else 'Đã tắt'
        st.write(f"• **Trạng thái Camera:** {status_icon} {status_text}")

if __name__ == "__main__":
    main()