from flask import Flask, jsonify,request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from keras.models import load_model
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
model = load_model("model_aiang1.h5")
data = pd.read_csv('dataset_1.csv')
df = pd.DataFrame(data)
vectorizer = CountVectorizer()
nama_encoded = vectorizer.fit_transform(df['nama_kegiatan']).toarray()

label_encoder_kategori = LabelEncoder()
kategori_encoded = label_encoder_kategori.fit_transform(df['kategori'])

label_encoder_prioritas = LabelEncoder()
prioritas_encoded = label_encoder_prioritas.fit_transform(df['prioritas'])

X = pd.concat([pd.DataFrame(nama_encoded), pd.Series(kategori_encoded), pd.Series(prioritas_encoded)], axis=1)

  # Encode label
label_encoder_nama = LabelEncoder()
# y = label_encoder_nama.fit_transform(df['nama_kegiatan'])
@app.route("/")
def index():
    return jsonify({
        "status":{
            "code":200,
            "message":"Success fetching the API",
        },
        "data":None
    }),200

@app.route("/prediction",methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        data = request.get_json(force=True)
        nama_kegiatan = data['nama_kegiatan']
        kategori = data['kategori']
        prioritas = data['prioritas']

        # Fit label encoders with the necessary data
        data_for_fitting = pd.read_csv('dataset_1.csv')
        vectorizer.fit(data_for_fitting['nama_kegiatan'])
        label_encoder_kategori.fit(data_for_fitting['kategori'])
        label_encoder_prioritas.fit(data_for_fitting['prioritas'])
        label_encoder_nama.fit(data_for_fitting['nama_kegiatan'])

        # Preprocess input data
        nama_encoded = vectorizer.transform([nama_kegiatan]).toarray()
        kategori_encoded = label_encoder_kategori.transform([kategori])[0]  # Use [0] to get the first element
        prioritas_encoded = label_encoder_prioritas.transform([prioritas])[0]  # Use [0] to get the first element

        # Combine the encoded features
        X_input = pd.concat([pd.DataFrame(nama_encoded), pd.Series([kategori_encoded]), pd.Series([prioritas_encoded])], axis=1)

        # Make prediction using the loaded model
        prediction = model.predict(X_input)
        predicted_name = label_encoder_nama.inverse_transform(prediction.argmax(axis=1))[0]

        # new_df=pd.DataFrame(json_)
        # nama_encoded_new = vectorizer.transform(new_df['nama_kegiatan']).toarray()
        # kategori_encoded_new = label_encoder_kategori.transform(new_df['kategori'])
        # prioritas_encoded_new = label_encoder_prioritas.transform(new_df['prioritas'])
        # X_new = pd.concat([pd.DataFrame(nama_encoded_new), pd.Series(kategori_encoded_new), pd.Series(prioritas_encoded_new)], axis=1)
        # predictions = model.predict(X_new)
        # predicted_names = label_encoder_nama.inverse_transform(predictions.argmax(axis=1))
        # print(predicted_names)
        return jsonify({'Prediction Task':(predicted_name)})
    else:
        return jsonify({
            "status":{
                "code":405,
                "message":"invalid method"
            },
            "data":None,
        }), 405
if __name__ == "__main__":
    app.run(port=3000)
    