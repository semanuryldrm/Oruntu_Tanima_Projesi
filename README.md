# 🇹🇷 Gerçek Zamanlı Örüntü Tanıma ve Duygu Analizi
### (Real-Time Pattern Recognition & Sentiment Analysis)

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

> **Bilgisayar Mühendisliği | Örüntü Tanıma Dersi Final Projesi**

Bu proje, Türkçe metinleri analiz ederek **Pozitif**, **Negatif** veya **Nötr** duygu durumlarını sınıflandıran makine öğrenimi tabanlı bir sistemdir. Twitter veri setleri üzerine inşa edilen model, **Data Augmentation** ve **Sentetik Veri Enjeksiyonu** teknikleri ile optimize edilmiştir.

---

## 🚀 Proje Özellikleri

Model, her oturumda veriyi dinamik olarak işleyen bir **Canlı Eğitim (Live Training)** mimarisine sahiptir.

* **🧠 Hibrit Veri Seti:** İki farklı geniş kapsamlı veri seti birleştirilerek veri çeşitliliği artırılmıştır.
* **💉 Veri Artırma (Data Augmentation):**
    * **Nötr Sınıfı:** Bilimsel ve coğrafi gerçekler eklenerek modelin bilgi cümlelerini "şikayet" sanması engellenmiştir.
    * **Pozitif Sınıfı:** "Ders", "Sınav" gibi akademik kelimelerin başarı bağlamındaki örüntüleri öğretilmiştir.
* **📊 N-Grams Analizi:** Kelimeler tek tek değil, ikili gruplar (Bigrams) halinde analiz edilerek bağlam kaybı önlenmiştir.
* **🎨 İnteraktif UI:** Streamlit framework'ü ile geliştirilen kullanıcı dostu arayüz.

---

## 🛠️ Teknoloji Yığını (Tech Stack)

* **🐍 Python 3.11**
* **🐼 Pandas:** Veri manipülasyonu ve temizleme.
* **🤖 Scikit-Learn:** Multinomial Naive Bayes algoritması.
* **🎨 Streamlit:** Web tabanlı arayüz geliştirme.

---



---

## 🧠 Algoritma ve Çalışma Mantığı

Sistem, metin sınıflandırma problemlerinde yüksek doğruluk ve hız sunan **Multinomial Naive Bayes** algoritmasını temel alır.

1.  **Ön İşleme:** Metinler küçük harfe çevrilir, linkler, sayılar ve noktalama işaretleri RegEx ile temizlenir.
2.  **Vektörleştirme (TF-IDF):** Kelimelerin metin içindeki önemi matematiksel olarak ağırlıklandırılır.
3.  **Dengeleme:** Eğitim sırasında sınıflar arası sayısal dengesizlik (Imbalance) giderilerek tarafsız bir tahmin mekanizması oluşturulur.

---

## ⚙️ Kurulum ve Kullanım

Projeyi yerelinizde çalıştırmak için:

1.  Repoyu klonlayın: `git clone https://github.com/semanuryldrm/Oruntu_Tanima_Projesi.git`
2.  Gerekli kütüphaneleri yükleyin: `pip install pandas scikit-learn streamlit`
3.  Uygulamayı başlatın: `streamlit run app.py`

---

## 📂 Dosya Yapısı

* `app.py`: Ana uygulama ve model eğitim motoru.
* `sentimentSet.csv`: Duygu analizi veri seti.
* `Türkçe Tweetlerde Analiz(Etiketli).csv`: Etiketlenmiş sosyal medya verisi.

---

**👤 Geliştiren:** Semanur Yıldırım
