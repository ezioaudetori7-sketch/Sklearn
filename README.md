import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import arabic_reshaper
from bidi.algorithm import get_display

# ========== 1. تحميل البيانات ==========
df = pd.read_csv('tweets.csv')
texts = df['text'].astype(str).tolist()
labels = df['label'].tolist()

# ========== 2. ترميز المشاعر ==========
le = LabelEncoder()
y = le.fit_transform(labels)

# ========== 3. تحويل النصوص إلى أرقام ==========
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
seq = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(seq, maxlen=10, padding='post')

# ========== 4. بناء النموذج ==========
model = Sequential([
    Embedding(5000, 32, input_length=10),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(set(y)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# ========== 5. تدريب النموذج ==========
model.fit(padded, np.array(y), epochs=20, verbose=1)

# ========== 6. تجربة تنبؤ جديد ==========
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=10, padding='post')
    pred = model.predict(pad)
    label_index = np.argmax(pred)
    label = le.inverse_transform([label_index])[0]
    return label, float(np.max(pred))

# ========== 7. اختبار النموذج ==========
tests = [
    "الخدمة ممتازة وسريعة",
    "أنا زعلان جدًا من التحديث",
    "ما في شي جديد اليوم"
]

for t in tests:
    reshaped_text = arabic_reshaper.reshape(t)
    bidi_text = get_display(reshaped_text)
    label, prob = predict_sentiment(t)
    print(f"\n📘 النص: {bidi_text}")
    print(f"🔹 النتيجة: {label} ({round(prob,2)})")
