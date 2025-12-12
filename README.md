# 🇹🇷 Gerçek Zamanlı Örüntü Tanıma ve Duygu Analizi
### (Real-Time Pattern Recognition & Sentiment Analysis)

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **Bilgisayar Mühendisliği | Örüntü Tanıma Dersi Final Projesi**

Bu proje, Türkçe metinleri analiz ederek **Pozitif**, **Negatif** veya **Nötr** duygu durumlarını (örüntülerini) sınıflandıran, makine öğrenimi tabanlı interaktif bir web uygulamasıdır. Klasik sınıflandırma yöntemlerinin aksine, **Data Augmentation (Veri Artırma)**, **Sentetik Veri Enjeksiyonu** ve **Hibrit Eğitim** teknikleri kullanılarak modelin "Kelime Yanlılığı" (Domain Bias) ve "Veri Dengesizliği" problemleri çözülmüştür.

---

## 🚀 Projenin Farkı ve Özellikleri

Bu sistem sadece statik bir modeli kullanmaz; başlatıldığı anda verileri işleyerek **gerçek zamanlı (live)** eğitim yapar.

### 1. 🧠 Hibrit ve Canlı Eğitim
İki farklı veri seti (`sentimentSet` ve `Türkçe Tweetler`) birleştirilerek geniş bir kelime havuzu oluşturulur ve model her başlatıldığında sıfırdan eğitilir.

### 2. 💉 Sentetik Veri Enjeksiyonu (Data Augmentation)
Modelin yanlış öğrendiği veya veri setinde eksik olan örüntüler, sentetik verilerle desteklenmiştir:
* **Nötr Sınıfı İyileştirmesi:** Coğrafi, bilimsel ve günlük hayat gerçekleri ("Ankara başkenttir", "Su 100 derecede kaynar") eklenerek modelin bilgi cümlelerini "şikayet" sanması engellendi.
* **Pozitif Sınıfı İyileştirmesi:** Öğrenciler için genelde negatif olan "Ders", "Sınav", "Proje" kelimelerinin başarı bağlamındaki kullanımları ("Dersi kavramış", "Sınavı geçti") modele öğretildi.
* **Negatif Sınıfı İyileştirmesi:** Eksik olan argo ve memnuniyetsizlik kalıpları güçlendirildi.

### 3. 🚫 Zehirli Kelime Filtresi (Stop Words)
Modeli yanıltan, ironi içeren veya bağlamdan kopuk kelimeler (Örn: "Güldüm", "haha" gibi şikayet cümlelerinde geçebilen kelimeler) özel bir filtre ile elendi.

### 4. 📊 N-Grams (Bigram) Analizi
Model sadece tek kelimelere değil, kelime gruplarına (Örn: "Güzel değil", "Terk ettim") bakarak bağlamı anlar.

---

## 🛠️ Teknoloji Yığını (Tech Stack)

| Bileşen | Teknoloji | Açıklama |
| :--- | :--- | :--- |
| **Backend** | Python 3.11+ | Ana programlama dili |
| **Arayüz (UI)** | Streamlit | Web tabanlı interaktif arayüz |
| **ML Algoritması** | Scikit-Learn | Multinomial Naive Bayes |
| **Vektörleştirme** | TF-IDF | Bigram (1-2 kelime) analizi |
| **Veri İşleme** | Pandas | Veri temizleme ve manipülasyon |

---

## 📸 Ekran Görüntüleri

*(Buraya uygulamanızın ekran görüntüsünü ekleyebilirsiniz)*

---

## ⚙️ Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

### 1. Repoyu Klonlayın
```bash
git clone [https://github.com/KULLANICI_ADINIZ/PROJE_ADINIZ.git](https://github.com/KULLANICI_ADINIZ/PROJE_ADINIZ.git)
cd PROJE_ADINIZ
