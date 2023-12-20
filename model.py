import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Contoh data (gantilah dengan data sesuai kebutuhan Anda)
# data = [
#     {'nama_kegiatan': 'Bermain game', 'kategori': 'fun', 'prioritas': 'high'},
#     {'nama_kegiatan': 'Lari pagi', 'kategori': 'exercise', 'prioritas': 'high'},
#     {'nama_kegiatan': 'nongkrong', 'kategori': 'fun', 'prioritas': 'low'},
#     {'nama_kegiatan': 'valorant', 'kategori': 'fun', 'prioritas': 'low'},
#     {'nama_kegiatan': 'Belajar matematika', 'kategori': 'study', 'prioritas': 'low'},
#     # ...
# ]
def model_aiang():


  data = pd.read_csv('dataset_1.csv')

  # Konversi data menjadi DataFrame

  df = pd.DataFrame(data)

  # Preprocessing data
  vectorizer = CountVectorizer()
  nama_encoded = vectorizer.fit_transform(df['nama_kegiatan']).toarray()

  label_encoder_kategori = LabelEncoder()
  kategori_encoded = label_encoder_kategori.fit_transform(df['kategori'])

  label_encoder_prioritas = LabelEncoder()
  prioritas_encoded = label_encoder_prioritas.fit_transform(df['prioritas'])

  # Gabungkan fitur-fitur yang telah diencode
  X = pd.concat([pd.DataFrame(nama_encoded), pd.Series(kategori_encoded), pd.Series(prioritas_encoded)], axis=1)

  # Encode label
  label_encoder_nama = LabelEncoder()
  y = label_encoder_nama.fit_transform(df['nama_kegiatan'])

  # Bagi data menjadi data latih dan data uji
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Buat model menggunakan Keras
  model = keras.Sequential([
      keras.layers.Dense(128, input_dim=X_train.shape[1], activation='relu'),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(64, activation='relu'),
      keras.layers.Dense(len(label_encoder_nama.classes_), activation='softmax')
  ])

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # Latih model
  model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

  # Evaluasi model
  # accuracy = model.evaluate(X_test, y_test)[1]
  # print(f'Akurasi model: {accuracy}')

  # Prediksi kegiatan baru
  new_data = [
      {'nama_kegiatan': 'gym', 'kategori': 'exercise', 'prioritas': 'high'},

      # ...
  ]

  new_df = pd.DataFrame(new_data)
  nama_encoded_new = vectorizer.transform(new_df['nama_kegiatan']).toarray()
  kategori_encoded_new = label_encoder_kategori.transform(new_df['kategori'])
  prioritas_encoded_new = label_encoder_prioritas.transform(new_df['prioritas'])

  X_new = pd.concat([pd.DataFrame(nama_encoded_new), pd.Series(kategori_encoded_new), pd.Series(prioritas_encoded_new)], axis=1)

  predictions = model.predict(X_new)
  predicted_names = label_encoder_nama.inverse_transform(predictions.argmax(axis=1))
  print(f'Prediksi kegiatan baru: {predicted_names}')
  return model

if __name__ == '__main__':
    model = model_aiang()
    model.save("model_aiang1.h5")