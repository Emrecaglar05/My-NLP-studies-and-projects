# Amazon Reviews Sentiment Analysis

Bu proje, Amazon ürün yorumlarını analiz ederek duygu (sentiment) tahmini yapmak amacıyla hazırlanmıştır. Projede **metin ön işleme, görselleştirme ve makine öğrenmesi modelleri** kullanılmıştır.

---

## 📂 Proje İçeriği

- **amazon.xlsx**: Örnek Amazon yorumlarının bulunduğu veri seti.  
- **main.py** (veya proje kod dosyanız): Python ile veri ön işleme, görselleştirme ve makine öğrenmesi modeli oluşturma.  
- **README.md**: Projenin açıklamaları ve kullanım rehberi.

---

## 🛠 Kullanılan Kütüphaneler

- `pandas` → Veri okuma ve veri çerçevesi işlemleri  
- `numpy` → Sayısal işlemler  
- `matplotlib` → Grafik çizimi  
- `wordcloud` → Kelime bulutu görselleştirme  
- `nltk` → Doğal dil işleme ve stopwords  
- `textblob` → Kelime köklemeleri (lemmatization)  
- `sklearn` → Makine öğrenmesi modelleri ve veri vektörizasyonu  
- `warnings` → Uyarıları gizleme

> Not: Gerekli nltk paketleri için:  
```python
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
