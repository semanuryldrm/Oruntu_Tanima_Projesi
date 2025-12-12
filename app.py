import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .main-title {
        text-align: center;
        color: #2E86C1;
        font-family: 'Helvetica', sans-serif;
    }
    .sub-title {
        text-align: center;
        color: #5D6D7E;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. YAN MENÜ (DÜZELTİLDİ: ALT ALTA LİSTE) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=100)
    st.title("Proje Künyesi")
    st.info("**Ders:** Örüntü Tanıma")
    st.write("**Konu:** Türkçe Tweetlerde Duygu Analizi")
    st.write("---")
    
    st.markdown("### ⚙️ Teknoloji Yığını")
    # BURAYI DÜZELTTİM: Alt alta maddeler halinde
    st.markdown("""
    * 🐍 **Python 3.11**
    * 🐼 **Pandas** (Veri İşleme)
    * 🤖 **Scikit-Learn** (Yapay Zeka)
    * 🎨 **Streamlit** (Arayüz)
    """)
    
    st.write("---")
    st.markdown("### 🧠 Model Mimarisi")
    st.success("Algoritma: **Multinomial Naive Bayes**")
    st.warning("Teknik: **TF-IDF (Bigram)**")
    st.error("Eğitim: **Full Hibrit (3 Sınıf Sentetik)**")

# --- 3. MODELİ HAZIRLA ---
yasakli_kelimeler = [
    "güldüm", "haha", "hahaha", "jsjsjs", "lol", 
    "ya", "şey", "bir", "bu", "şu", "o", "ben", "sen",
    "kadar", "gibi", "için", "diye", "gidip"
]

# --- SENTETİK VERİ SETLERİ ---

# A) SENTETİK NÖTRLER
sentetik_notrler = [
    "toplantı yarın saat 14:00'te yapılacak", "bugün hava durumu parçalı bulutlu",
    "yarın okula gideceğim", "markete gidip ekmek alacağım",
    "otobüs durağında bekliyorum", "akşam yemeği için makarna yaptım",
    "telefonumun şarjı bitti", "kitap okuyorum", "televizyon izliyorum",
    "bilgisayar başında çalışıyorum", "sınav haftası başladı", "ders çalışmam lazım",
    "bugün günlerden salı", "hava biraz soğuk", "kargo paketim geldi",
    "sipariş durumu kargoda", "banka hesabı açtırdım", "doktordan randevu aldım",
    "türkiye'nin başkenti ankara'dır", "istanbul en kalabalık şehirdir",
    "nüfusu 5 milyondan fazladır", "coğrafya dersinde bölgeleri işledik",
    "türkiye bir yarımadadır", "su 100 derecede kaynar",
    "dünya güneş etrafında döner", "matematik sınavı zor değildi"
] * 20

# B) SENTETİK NEGATİFLER
sentetik_negatifler = [
    "film o kadar sıkıcıydı ki yarısında çıktım",
    "çok sıkıcı bir gündü hiç keyif alamadım",
    "mekanı terk ettim çünkü çok kötüydü",
    "ortam o kadar gergindi ki terk ettim",
    "bu ürün tam bir hayal kırıklığı",
    "beklediğimden çok daha kötü çıktı",
    "hiç beğenmedim param boşa gitti",
    "servis rezaletti bir daha asla gitmem",
    "tadı iğrençti midem bulandı",
    "bu ne biçim hizmet, yazıklar olsun"
] * 20

# C) SENTETİK POZİTİFLER
sentetik_pozitifler = [
    "tamam çocuk dersin adını ve içeriğini kavramış",
    "konuyu çok iyi anladım ve kavradım",
    "öğrenci dersi başarıyla geçti tebrikler",
    "projenin mantığını hemen kavramış",
    "dersin içeriği çok zengin ve öğreticiydi",
    "çocuklar konuyu hemen anladı harikalar",
    "bu derste çok şey öğrendim teşekkürler",
    "sınavdan yüksek not aldım çok mutluyum",
    "başarılı bir çalışma olmuş eline sağlık",
    "tamamdır bu iş olmuş gayet güzel",
    "anlatılan her şeyi eksiksiz kavramış",
    "performansı gayet yerinde tebrik ediyorum"
] * 30

@st.cache_resource
def modeli_egit():
    try:
        # Dosya 1
        df1 = pd.read_csv('Türkçe Tweetlerde Analiz(Etiketli).csv', encoding='utf-8')
        df1.dropna(subset=['Tweet'], inplace=True)
        map1 = {'Negatif': 0, 'Nötr': 1, 'Pozitif': 2}
        df1['label'] = df1['Etiket'].map(map1)
        df1 = df1[['Tweet', 'label']].rename(columns={'Tweet': 'text'})

        # Dosya 2
        df2 = pd.read_csv('sentimentSet.csv', encoding='utf-8')
        df2.dropna(subset=['tweets'], inplace=True)
        map2 = {-1: 0, 0: 1, 1: 2}
        df2['label'] = df2['sentiment'].map(map2)
        df2 = df2[['tweets', 'label']].rename(columns={'tweets': 'text'})

        df_final = pd.concat([df1, df2], ignore_index=True)
        
        def temizle(text):
            text = str(text).lower()
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            return text
        
        df_final['clean_text'] = df_final['text'].apply(temizle)
        df_final.dropna(subset=['label'], inplace=True)
        df_final['label'] = df_final['label'].astype(int)
        
        # TÜM SENTETİK VERİLERİ BİRLEŞTİR
        df_synth_neu = pd.DataFrame({'clean_text': sentetik_notrler, 'label': 1})
        df_synth_neg = pd.DataFrame({'clean_text': sentetik_negatifler, 'label': 0})
        df_synth_pos = pd.DataFrame({'clean_text': sentetik_pozitifler, 'label': 2})
        
        df_final = pd.concat([df_final, df_synth_neu, df_synth_neg, df_synth_pos], ignore_index=True)

    except FileNotFoundError:
        st.error("HATA: CSV dosyaları bulunamadı!")
        st.stop()

    # Dengeleme
    min_sayi = df_final['label'].value_counts().min()
    df_dengeli = pd.concat([
        df_final[df_final['label'] == 0].sample(n=min_sayi, random_state=42),
        df_final[df_final['label'] == 1].sample(n=min_sayi, random_state=42),
        df_final[df_final['label'] == 2].sample(n=min_sayi, random_state=42)
    ])

    # Eğitim
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words=yasakli_kelimeler)
    X_vec = vectorizer.fit_transform(df_dengeli['clean_text'])
    y = df_dengeli['label']
    model = MultinomialNB()
    model.fit(X_vec, y)
    
    return model, vectorizer

# Modeli Sessizce Yükle
with st.spinner('Sistem Başlatılıyor: Veriler Okunuyor ve Model Anlık Olarak Eğitiliyor...'):
    model, vectorizer = modeli_egit()

# --- 4. ANA EKRAN TASARIMI ---
st.markdown("<h1 class='main-title'>Gerçek Zamanlı Örüntü Tanıma ve Duygu Analizi</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Hibrit Veri ile Eğitilen Dinamik NLP Modeli</p>", unsafe_allow_html=True)
st.write("---")

if 'metin' not in st.session_state:
    st.session_state['metin'] = ""

col_input, col_result = st.columns([1.5, 1])

with col_input:
    st.subheader("📝 Metin Girişi")
    
    st.markdown("**Hızlı Test Verisi:**")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    if btn_col1.button("😡 Negatif"):
        st.session_state['metin'] = "Film o kadar sıkıcıydı ki yarısında salonu terk ettim."
    if btn_col2.button("😐 Nötr"):
        st.session_state['metin'] = "Türkiye'nin başkenti Ankara'dır ve nüfusu 5 milyondan fazladır."
    if btn_col3.button("😊 Pozitif"):
        st.session_state['metin'] = "Tamam, çocuk dersin adını ve içeriğini kavramış."
        
    tweet_input = st.text_area("Analiz kutusu:", value=st.session_state['metin'], height=150, placeholder="Analiz edilecek metni buraya giriniz...")
    analyze_btn = st.button("🚀 ANALİZ ET", type="primary")

with col_result:
    st.subheader("📊 Örüntü Sonucu")
    
    if analyze_btn and tweet_input:
        text = str(tweet_input).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        vektor = vectorizer.transform([text])
        tahmin = model.predict(vektor)[0]
        
        # Sonuç Kartları
        if tahmin == 2: # POZİTİF
            st.markdown("""
            <div class="result-card" style="background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb;">
                <h1>😊<br>POZİTİF</h1>
                <p>Tespit Edilen Örüntü: <b>Olumlu / Başarılı</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        elif tahmin == 0: # NEGATİF
            st.markdown("""
            <div class="result-card" style="background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb;">
                <h1>😡<br>NEGATİF</h1>
                <p>Tespit Edilen Örüntü: <b>Olumsuz / Şikayet</b></p>
            </div>
            """, unsafe_allow_html=True)
            
        else: # NÖTR
            st.markdown("""
            <div class="result-card" style="background-color: #fff3cd; color: #856404; border: 2px solid #ffeeba;">
                <h1>😐<br>NÖTR</h1>
                <p>Tespit Edilen Örüntü: <b>Durum Bildirimi / Bilgi</b></p>
            </div>
            """, unsafe_allow_html=True)
            
        with st.expander("🔍 Modelin Gördüğü İşlenmiş Veri"):
            st.code(text, language="text")

    elif analyze_btn:
        st.warning("Lütfen analiz edilecek bir metin giriniz.")
    else:
        st.info("Sistem hazır. Sol taraftan veri girişi yapabilirsiniz.")