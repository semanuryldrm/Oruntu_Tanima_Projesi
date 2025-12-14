# 🇹🇷 Gelişmiş Duygu Analizi ve Örüntü Tanıma Projesi
### (Advanced Sentiment Analysis & Pattern Recognition System)

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/AI-Naive%20Bayes-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **Bilgisayar Mühendisliği | Örüntü Tanıma Dersi Final Projesi**
>
> **Geliştirici:** Semanur Yıldırım

Bu proje, Türkçe metinler üzerindeki gizli örüntüleri tespit ederek **Pozitif**, **Negatif** ve **Nötr** duygu durumlarını sınıflandıran, yüksek başarı oranına sahip bir makine öğrenimi sistemidir.

Proje, klasik sınıflandırma yöntemlerinin ötesine geçerek; **"Zor Örnek Madenciliği (Hard Example Mining)"**, **"Random Swap Veri Artırma"** ve **"Bağlamsal Düzeltme"** teknikleriyle güçlendirilmiştir.

---

## 🚀 Projenin Öne Çıkan Özellikleri (Key Features)

### 1. 🧠 Zor Örnek Madenciliği (Hard Mining & Weighted Training)
Standart modellerin hata yaptığı karmaşık cümle yapıları için özel eğitim modülleri geliştirilmiştir. Bu özel verilere eğitim sırasında **50x ağırlık (weight)** verilerek modelin şu ince ayrımları yapması sağlanmıştır:

* **"Ama" Bağlacı Tuzakları:** * *Örnek:* "Ürünü büyük hevesle aldım **ama** hiç çalışmadı." 
  * *Sonuç:* Başındaki olumlu kelimelere ("heves", "aldım") aldanmayıp, sonundaki memnuniyetsizliği tespit eder (**Negatif**).
  
* **Kültür & Sanat Ayrımı:** * *Örnek:* "Bu kitap 19. yüzyıl Rus edebiyatını anlatır."
  * *Sonuç:* "Rus" veya "Kitap" kelimelerini sıkıcı/negatif olarak değil, ansiklopedik bilgi olarak tanır (**Nötr**).
  
* **Argo & Ters Köşe:** * *Örnek:* "Korkunç güzel bir filmdi." veya "Yıkılıyor ortalık."
  * *Sonuç:* Kelimelerin bağlam içindeki olumlu anlamlarını kavrar (**Pozitif**).

### 2. 🔄 Random Swap Veri Artırma (Data Augmentation)
Veri setindeki dengesizliği gidermek ve ezberlemeyi (overfitting) önlemek için **Random Swap** algoritması kullanılmıştır.
* Cümle içindeki kelimelerin yerleri rastgele değiştirilerek sentetik veriler üretilmiştir.
* Bu yöntem, dilin yapısını bozmadan modelin kelime ilişkilerini daha iyi öğrenmesini sağlar.

### 3. ⚖️ Tam Dengeli Sınıf Dağılımı
Başlangıçta dengesiz olan veri seti, veri artırma teknikleri ile her üç sınıf (Pozitif, Negatif, Nötr) için eşit sayıda örneğe tamamlanmıştır. Bu sayede modelin herhangi bir sınıfa yanlı (biased) davranması engellenmiştir.

### 4. ⚡ Optimize Edilmiş Mimari (Inference-Only)
Model her açılışta tekrar eğitilmez.
* Eğitim süreci arka planda tamamlanıp `.pkl` dosyası olarak kaydedilir.
* Uygulama (`app.py`), hazır eğitilmiş beyni yükler ve **milisaniyeler içinde** yanıt verir.

---

## 🛠️ Teknoloji Yığını (Tech Stack)

Projede kullanılan teknolojiler, üstlendikleri görevlere göre aşağıda listelenmiştir:

* **Programlama Dili ve Altyapı:**
    * **Python 3.13:** Projenin temel geliştirme ortamı olarak, dilin en güncel ve yüksek performanslı sürümü tercih edilmiştir.

* **Kullanıcı Arayüzü (UI/UX):**
    * **Streamlit:** Kullanıcı dostu, web tabanlı bir arayüz oluşturmak için kullanılmıştır. "Terminal Modu" ve özel renk paletleri için **Custom CSS** entegrasyonu yapılmıştır.

* **Makine Öğrenimi Algoritması:**
    * **Scikit-Learn (Multinomial Naive Bayes):** Metin sınıflandırma problemlerinde (özellikle kelime frekanslarına dayalı analizlerde) yüksek doğruluk ve hız sağladığı için bu algoritma seçilmiştir.

* **Veri Artırma ve İyileştirme (Data Augmentation):**
    * **Random Swap & Hard Mining:** Veri setindeki dengesizliği gidermek ve modelin "zor" cümleleri (ironi, bağlaçlar vb.) anlaması için özel sentetik veri üretme algoritmaları geliştirilmiştir.

* **Özellik Çıkarımı (Feature Extraction):**
    * **TF-IDF (Bigram Destekli):** Metinleri makinenin anlayacağı sayısal vektörlere dönüştürmek için kullanılmıştır. Tekli kelimeler yerine ikili kelime gruplarını (Bigram) da analiz ederek bağlam kaybını önler.

* **Veri Manipülasyonu ve Temizlik:**
    * **Pandas & NumPy:** Büyük veri setlerinin (.csv) okunması, birleştirilmesi, RegEx ile temizlenmesi ve matris işlemleri için kullanılmıştır.

* **Model Optimizasyonu ve Performans:**
    * **Joblib:** Eğitilen modelin ve vektörleştiricinin diske kaydedilip (serialization), uygulamanın her açılışında tekrar eğitim yapmadan milisaniyeler içinde çalışmasını sağlamak için kullanılmıştır.

---

## 🧠 Algoritma Akışı

1.  **Veri Entegrasyonu:** Farklı kaynaklardan gelen `.csv` veri setleri birleştirilir.
2.  **Hard Mining Enjeksiyonu:** Modelin kafasını karıştıran özel senaryolar manuel olarak veri setine yüksek ağırlıkla enjekte edilir.
3.  **Ön İşleme (Preprocessing):**
    * RegEx ile link, mention, noktalama işareti temizliği.
    * Küçük harfe dönüştürme (Case folding).
4.  **Veri Artırma:** Azınlık sınıfları için kelime karıştırma (Random Swap) ile sentetik veri üretimi.
5.  **Eğitim (Training):** TF-IDF vektörleri üzerinden Naive Bayes algoritması ile model eğitilir.
6.  **Tahmin (Prediction):** Kullanıcıdan gelen veri canlı olarak temizlenir ve sınıflandırılır.

---

## 🖥️ Arayüz Tasarımı

Proje, kullanıcı deneyimini artırmak için özel CSS ile tasarlanmış modern bir arayüze sahiptir:
* **Terminal Tarzı Veri Gösterimi:** Arka planda işlenen ham veriyi (Cleaned Data) koyu modda, kod bloğu şeklinde gösterir.
* **Dinamik Sonuç Kartları:** Tahmin sonucuna göre (Mutlu, Üzgün, Nötr) renk değiştiren ve gölgeli kart tasarımı.

---

## ⚙️ Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak için adımları takip edin:

**1. Gerekli Kütüphaneleri Yükleyin:**
```bash
pip install pandas scikit-learn streamlit joblib
```
**2. Modeli Eğitin (Opsiyonel):**
Eğer veri setinde değişiklik yaptıysanız, Jupyter Notebook dosyasını (`Örüntü_tanıma_proje.ipynb`) çalıştırarak `final_model.pkl` dosyasını güncelleyin. (Hazır dosyalar projede mevcuttur).

**3. Uygulamayı Başlatın:**
Terminal veya komut satırına şu kodu yazın:

```bash
streamlit run app.py
```

