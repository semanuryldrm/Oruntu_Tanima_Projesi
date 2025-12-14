import streamlit as st
import joblib
import re

# --- 1. SAYFA VE STİL AYARLARI ---
st.set_page_config(
    page_title="Duygu Analizi Projesi", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Tasarımı
st.markdown("""
<style>
    .result-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: scale(1.02);
    }
    .main-title {
        text-align: center;
        color: #1565C0;
        font-family: 'Helvetica', sans-serif;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-title {
        text-align: center;
        color: #546E7A;
        font-size: 18px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. MODEL VE VEKTÖRLEŞTİRİCİYİ YÜKLEME ---
@st.cache_resource
def model_yukle():
    try:
        model = joblib.load('final_model.pkl')
        vectorizer = joblib.load('final_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("HATA: 'final_model.pkl' bulunamadı. Lütfen önce eğitimi tamamlayın.")
        return None, None

model, vectorizer = model_yukle()

# Temizlik Fonksiyonu
def temizle_metin(metin):
    metin = str(metin).lower()
    metin = re.sub(r'http\S+|www\S+', '', metin)
    metin = re.sub(r'@[A-Za-z0-9]+', '', metin)
    metin = re.sub(r'[^\w\s]', '', metin)
    metin = re.sub(r'\d+', '', metin)
    return metin

# --- HTML OLUŞTURUCU FONKSİYON (HATAYI ÇÖZEN KISIM) ---
def get_terminal_html(text):
    # Bu fonksiyon HTML kodunu sıfır girinti ile oluşturur.
    # Böylece </div> hatası oluşmaz.
    return f"""
<div style="background-color: #1E1E1E; border-left: 6px solid #FFD700; border-radius: 10px; padding: 20px; margin-top: 20px; margin-bottom: 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.3); color: #E0E0E0;">
<div style="display: flex; align-items: center; margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px;">
<span style="font-size: 20px; margin-right: 10px;">⚙️</span>
<span style="font-weight: bold; color: #FFD700; font-family: monospace; letter-spacing: 1px;">ALGORİTMA GİRDİSİ (PROCESSED DATA)</span>
</div>
<div style="font-family: 'Courier New', monospace; color: #00FF7F; font-size: 15px; background-color: #000000; padding: 15px; border-radius: 5px;">
> {text}
</div>
</div>
"""

# --- 3. YAN MENÜ ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=100)
    st.title("Proje Hakkında")
    
    st.markdown("""
    ### 🎓 Örüntü Tanıma Dersi Final Projesi
    
    Bu çalışma, **Bilgisayar Mühendisliği Bölümü, Örüntü Tanıma** dersi kapsamında geliştirilmiştir.
    
    **Amaç:**
    Sosyal medya verileri üzerindeki gizli örüntüleri (pozitif, negatif, nötr) tespit edebilen bir yapay zeka modeli geliştirmektir.
    
    **Kullanılan Teknolojiler:**
    * 🐍 Python & Scikit-Learn
    * 🤖 Naive Bayes Algoritması
    * 📊 TF-IDF Vektörleştirme
    * 🔄 Veri Artırma (Data Augmentation)
    """)
    
    st.markdown("---")
    st.write("👩‍💻 Geliştirici: **Semanur Yıldırım**")

# --- 4. ANA EKRAN ---
st.markdown("<h1 class='main-title'>🧠 Gerçek Zamanlı Duygu Analizi</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Yapay Zeka Destekli Metin Sınıflandırma Modülü</p>", unsafe_allow_html=True)

# Text Input
text_input = st.text_area("Analiz edilecek metni buraya yazın:", height=130, placeholder="Örnek: Bu proje gerçekten beklentimin çok üzerinde, harika olmuş!")

# Buton
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    analiz_butonu = st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True)

# --- 5. ANALİZ VE SONUÇLAR ---
if analiz_butonu:
    if text_input and model:
        # 1. Temizlik
        clean_text = temizle_metin(text_input)
        
        # 2. Tahmin
        vektor = vectorizer.transform([clean_text])
        tahmin = model.predict(vektor)[0]
        
        # --- KESİN ÇÖZÜM: Fonksiyonu Çağırıyoruz ---
        st.markdown(get_terminal_html(clean_text), unsafe_allow_html=True)

        # --- SONUÇ KARTLARI ---
        col1, col2, col3 = st.columns([1,2,1])
        
        with col2:
            if tahmin == 2: # POZİTİF
                st.markdown("""
                <div class="result-card" style="background-color: #d1e7dd; color: #0f5132; border: 2px solid #badbcc;">
                    <h1 style="margin:0;">😊</h1>
                    <h2 style="margin:10px 0;">POZİTİF</h2>
                    <p style="font-size:14px;">Algılanan Duygu: <b>Mutluluk / Memnuniyet</b></p>
                </div>
                """, unsafe_allow_html=True)
                
            elif tahmin == 0: # NEGATİF
                st.markdown("""
                <div class="result-card" style="background-color: #f8d7da; color: #842029; border: 2px solid #f5c2c7;">
                    <h1 style="margin:0;">😡</h1>
                    <h2 style="margin:10px 0;">NEGATİF</h2>
                    <p style="font-size:14px;">Algılanan Duygu: <b>Öfke / Şikayet / Üzgünlük</b></p>
                </div>
                """, unsafe_allow_html=True)
                
            else: # NÖTR (1)
                st.markdown("""
                <div class="result-card" style="background-color: #fff3cd; color: #664d03; border: 2px solid #ffecb5;">
                    <h1 style="margin:0;">😐</h1>
                    <h2 style="margin:10px 0;">NÖTR</h2>
                    <p style="font-size:14px;">Algılanan Duygu: <b>Tarafsız / Bilgi İçerikli</b></p>
                </div>
                """, unsafe_allow_html=True)
                
    elif not text_input:
        st.warning("⚠️ Lütfen analiz etmek için bir şeyler yazın.")