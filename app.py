from flask import Flask,render_template,request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app=Flask(__name__)

UPLOAD_FOLDER="static/uploads"
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER

model=tf.keras.models.load_model("model.h5")

with open("labels.txt") as f:
    labels=[i.strip() for i in f.readlines()]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    file=request.files["image"]

    path=os.path.join(app.config["UPLOAD_FOLDER"],file.filename)
    file.save(path)

    img=image.load_img(path,target_size=(128,128))
    img=image.img_to_array(img)

    img=np.expand_dims(img,axis=0)

    img=img/255.0

    prediction=model.predict(img)

    index=np.argmax(prediction)

    result=labels[index]

    confidence=round(np.max(prediction)*100,2)

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        img=path
    )

if __name__=="__main__":
    app.run(debug=True)