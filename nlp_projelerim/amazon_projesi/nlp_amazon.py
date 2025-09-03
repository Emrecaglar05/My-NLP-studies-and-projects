import nltk
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from warnings import filterwarnings
import warnings

warnings.filterwarnings('ignore')                  # Uyarıları gizle
pd.set_option('display.max_columns', None)         # Tüm sütunları göster
pd.set_option('display.float_format', lambda x: '%.2f' % x)  # Float → 2 ondalık
pd.set_option('display.width', 200)               # Satır genisligini ayarla

df = pd.read_excel("amazon.xlsx")
print(df.head())
print(df.info())

# METIN ON ISLEME #

df["Review"] =  df["Review"].astype(str)
df["Review"] = df["Review"].str.lower() # Tüm metini küçük harfe çevirdik
df["Review"] = df["Review"].replace(r"[^\w\s]", " ", regex=True) # Noktalama ısaretlerını kaldırdık
df["Review"] = df["Review"].str.replace(r"\d+", " ", regex=True) # Sayıları kaldırdık

# Gereksiz kelımeler cıkarma ( stopwords )

 # 1 =  nltk.download('stopwords')
stop_words = set(stopwords.words("english"))  # Ingilice stopwords listesi
df["Review"] = df["Review"].apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))     #  apply → sütundaki veya satırdaki tüm veriye belirttiğin fonksiyonu uygular.

# Frekansı 5  den az olan kelimeleri metinden çıkarma
# nltk.download('wordnet')
all_words = pd.Series(" ".join(df["Review"]).split()) #> Tüm yorumları tek metın halıne getırır. Kelımelere ayırır ve pandas serısıne donusturur.
word_counts = all_words.value_counts()  # Kelimenın metınde kac defa gectıgını hesaplar.
rare_words = word_counts[word_counts < 5].index # 5 den kucuk olanları kaydeder.
df["Review"] = df["Review"].apply(lambda x: " ".join([w for w in x.split() if w not in rare_words]))  # 5 den kucuk olanları sıler

df["Review"] = df["Review"].apply(lambda x: " ".join([Word(w).lemmatize() for w in x.split()]))  # Kelımelerı köklerıne ayır

print(df["Review"].head(10))

## METIN GORSELLESTIRME ##

all_words = " ".join(df['Review']).split()      # Tüm yorumları birleştir ve kelimelere ayır
tf = pd.Series(all_words).value_counts().reset_index()  # Kelime frekanslarını hesapla

tf.columns = ['words', 'tf']  # 'index' -> 'words', 0 -> 'tf'
print(tf.head(10))

tf_filtered = tf[tf['tf'] > 500]

plt.figure(figsize=(12,6))
plt.bar(tf_filtered['words'], tf_filtered['tf'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title("Kelime Frekansları (TF > 500)")
plt.xlabel("Kelime")
plt.ylabel("Frekans")
plt.show()


# KELIME BULUTU OLUSTURMA (WORDCLOUD)
text = " ".join(i for i in df.Review )

wordcloud =  WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# DUYGU ANALIZI #

sia = SentimentIntensityAnalyzer()

r_first_Ten  = df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)) #  Ilk 10 gözlem için polarity_scores() hesapla
print(r_first_Ten)

compound = df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"]) # # compound > 0 ise genel duygu pozitiftir, bu filtre pozitif yorumları seçer

print(compound)

df["Review"][0:10].apply(lambda x: "pos" if  sia.polarity_scores(x)["compound"] > 0 else "neg") # Compound skorlarına göre pos-neg ataması yap

df["Sentıment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg") #Tüm veri seti için sentiment değişkeni ekle

print(df.groupby("Sentıment_Label")["Star"].mean())

# MAKINE OGRENMESI MODELI #

# Bagımlı ve bagımsız degıskenlerımızı belırleyerek traın-test olarak ayıralım

train_x, test_x, train_y, test_y = train_test_split(df["Review"], df["Sentıment_Label"], random_state=42)

tf_idf_word_vectorizer = TfidfVectorizer().fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

log_model = LogisticRegression()
log_model.fit(x_train_tf_idf_word, train_y)

y_pred = log_model.predict(x_test_tf_idf_word)

print(classification_report(y_pred, test_y))

oran = cross_val_score(log_model, x_test_tf_idf_word, test_y, cv=5).mean()
print(oran)


# Örnek yapalım

# Örnek yapalım (TF-IDF ile)
yeni_yorum = df["Review"].sample(1).values
yeni_yorum_vec = tf_idf_word_vectorizer.transform(yeni_yorum)  # ✅ aynı vektörizer
pred = log_model.predict(yeni_yorum_vec)

print(f"Review: {yeni_yorum[0]} \nPrediction: {pred[0]}")

rf_model = RandomForestClassifier().fit(x_train_tf_idf_word, train_y)
oran = cross_val_score(rf_model, x_test_tf_idf_word, test_y, cv=5, n_jobs=1).mean()
print(oran)





