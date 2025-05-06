import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Báo cáo cuối kỳ",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, distinct styling
st.markdown("""
    <style>
        .stApp {
            background-color: #0f172a;
            font-family: 'Poppins', sans-serif;
            color: #f1f5f9;
        }

        /* Top navigation bar */
        .nav-bar {
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            padding: 15px 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }

        .nav-title {
            font-size: 1.8em;
            font-weight: 700;
            color: #ffffff;
            margin: 0;
        }

        .nav-links a {
            color: #e0e7ff;
            text-decoration: none;
            margin: 0 15px;
            font-size: 1.1em;
            transition: color 0.3s ease;
        }

        .nav-links a:hover {
            color: #facc15;
        }

        /* Hero section with animation */
        .hero-section {
            background: url('https://www.transparenttextures.com/patterns/stardust.png'), linear-gradient(135deg, #1e40af, #6d28d9);
            border-radius: 15px;
            padding: 50px;
            text-align: center;
            margin-bottom: 30px;
            animation: fadeIn 1.5s ease-in-out;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        }

        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        .hero-title {
            font-size: 3.2em;
            font-weight: 800;
            color: #f1f5f9;
            margin: 0;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.3);
        }

        .hero-subtitle {
            font-size: 1.4em;
            color: #d1d5db;
            margin-top: 15px;
        }

        /* Content grid */
        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 0 20px;
        }

        .content-card {
            background: #1e293b;
            border-radius: 12px;
            padding: 25px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .content-card:hover {
            background: #334155;
            transform: scale(1.05);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }

        .content-card h3 {
            color: #facc15;
            font-size: 1.8em;
            margin-bottom: 15px;
            font-weight: 600;
        }

        .content-card p, .content-card li {
            color: #d1d5db;
            font-size: 1.15em;
            line-height: 1.6;
        }

        /* Table of contents list */
        .toc-list {
            list-style: none;
            padding: 0;
        }

        .toc-list li {
            padding: 10px 0;
            display: flex;
            align-items: center;
            transition: color 0.3s ease;
        }

        .toc-list li:hover {
            color: #facc15;
        }

        .toc-list li::before {
            content: "✨";
            margin-right: 10px;
            color: #facc15;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .content-grid {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 2.5em;
            }

            .nav-bar {
                flex-direction: column;
                text-align: center;
            }

            .nav-links a {
                margin: 10px 0;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Top Navigation Bar
st.markdown("""
    <div class="nav-bar">
        <h1 class="nav-title">Báo cáo cuối kỳ</h1>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#student-info">Student Info</a>
            <a href="#toc">Table of Contents</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">Báo cáo cuối kỳ</h1>
        <p class="hero-subtitle">Khám phá bài báo cáo với giao diện hiện đại và trực quan! 🌟</p>
    </div>
""", unsafe_allow_html=True)

# Content Grid
st.markdown('<div class="content-grid">', unsafe_allow_html=True)

# Student Info Card
st.markdown('<div class="content-card" id="student-info">', unsafe_allow_html=True)
st.markdown("""
    <h3>📚 Thông tin sinh viên</h3>
    <p>🎓 <strong>Lại Hoàng Phúc Khải</strong> - 22110316</p>
    <p>🎓 <strong>Sinh viên thứ 2</strong> - Mã số sinh viên</p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Table of Contents Card
st.markdown('<div class="content-card" id="toc">', unsafe_allow_html=True)
st.markdown('<h3>📋 Mục lục</h3>', unsafe_allow_html=True)
st.markdown("""
    <ul class="toc-list">
        <li>Nhận diện trái cây</li>
        <li>Nhận diện khuôn mặt</li>
        <li>Xử lý ảnh (Chương 3, 4, 9)</li>
    </ul>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)