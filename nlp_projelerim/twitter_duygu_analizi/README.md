# 🐦 Twitter Duygu-Durum Analizi Projesi

Bu proje, **tweet'ler üzerinde duygu analizi** yaparak **pozitif, negatif veya nötr duyguları** sınıflandırmayı amaçlar.  
Proje, **Python tabanlı bir veri ön işleme pipeline'ı** ve **Logistic Regression** modeli kullanarak tweet'lerin duygu etiketlerini tahmin eder.  
Ayrıca, **VADER duygu analiz aracı** ile tweet'lere otomatik etiketleme yapılır ve **tarih-saat verilerinden ek özellikler çıkarılır**.

---

## ✨ Özellikler

- **Veri Ön İşleme:** Tweet'lerin temizlenmesi, tarih-saat dönüşümleri ve özellik çıkarımı  
- **Duygu Analizi:** VADER ile otomatik duygu etiketi atama (**pozitif, negatif, nötr**)  
- **Makine Öğrenimi:** TF-IDF vektörleştirme ve Logistic Regression ile sınıflandırma  
- **Örnek Tahmin:** Rastgele bir tweet seçip duygu tahmini yapma  

---

## 🛠️ Kullanılan Teknolojiler

- **Python 3.x**  
- **Kütüphaneler:**  
  - `pandas`: Veri işleme ve analiz  
  - `nltk`: VADER duygu analizi  
  - `sklearn`: Makine öğrenimi (TF-IDF, Logistic Regression, LabelEncoder)  

- **Veri Seti:** `tweets_labeled.csv` (tweet ve tarih içeren bir CSV dosyası)

















