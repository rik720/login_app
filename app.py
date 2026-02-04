from flask import Flask, render_template, request, jsonify
import face_recognition
import cv2
import numpy as np
import base64
import os 

app = Flask(__name__)

print("Loading reference image...")
ref_img = face_recognition.load_image_file("reference.jpg")
ref_encodings = face_recognition.face_encodings(ref_img)

if not ref_encodings:
    raise Exception("NO FACE FOUND IN reference.jpg")

ref_encoding = ref_encodings[0]
print("Reference face loaded")

@app.route("/")
def login():
    return render_template("login.html")

@app.route("/verify", methods=["POST"])
def verify():
    print("Verify endpoint hit")

    data = request.json.get("image")
    if not data:
        print("No image data received")
        return jsonify(success=False)

    img_bytes = base64.b64decode(data.split(",")[1])
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)

    faces = face_recognition.face_encodings(frame)
    print("Faces detected:", len(faces))

    if not faces:
        return jsonify(success=False)

    match = face_recognition.compare_faces(
        [ref_encoding], faces[0], tolerance=0.45
    )[0]

    print("Match result:", match)
    return jsonify(success=bool(match))

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))