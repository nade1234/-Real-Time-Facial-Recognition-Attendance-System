import streamlit as st

st.set_page_config(page_title='Attendance System', layout='wide')

# Custom CSS for glassmorphism and modern layout
st.markdown("""
    <style>
    body, .main, .block-container {background: #141824 !important;}
    .welcome-card {
        background: rgba(255,255,255,0.15);
        border-radius: 32px;
        padding: 56px 44px 36px 44px;
        max-width: 680px;
        margin: 70px auto 36px auto;
        box-shadow: 0 10px 30px 0 rgba(30,34,90,0.10);
        backdrop-filter: blur(9px);
        border: 1px solid rgba(255,255,255,0.28);
        text-align: center;
    }
    .welcome-title {
        font-size: 2.7rem;
        font-weight: 700;
        color: #fff;
        margin-bottom: 18px;
        letter-spacing: 1px;
    }
    .welcome-subtitle {
        font-size: 1.38rem;
        font-weight: 400;
        color: #e5e7ef;
        margin-bottom: 0.9rem;
    }
    .welcome-desc {
        font-size: 1.08rem;
        color: #c8cbda;
        margin-bottom: 1.8rem;
        line-height: 1.6;
        font-weight: 400;
    }
    .welcome-img {
        width: 250px;
        border-radius: 18px;
        margin: 18px auto 0 auto;
        box-shadow: 0 6px 16px rgba(30,34,90,0.13);
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">
            👋 WELCOME
        </div>
        <div class="welcome-subtitle">
            ENICAR Attendance System with Face Recognition
        </div>
        <div class="welcome-desc">
            This innovative system automates the attendance management of students and teachers using advanced facial recognition technology, ensuring efficiency, speed, and security.
        </div>
        <img class="welcome-img"
            src="https://img.freepik.com/vecteurs-libre/technologie-controle-identite-homme-illustration-dessin-anime_40876-16512.jpg"
            alt="Facial Recognition ENICAR"/>
    </div>
""", unsafe_allow_html=True)
