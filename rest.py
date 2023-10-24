from flask import Flask, request, jsonify
import tensorflow as tf
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def root():
    return jsonify(tf.__version__)

@app.route("/version")
def version():
    return "hello"

# @app.route("/version", methods = ['POST'])
# def hough():
#     if "image" in request.files:
#         b64 = request.files['image'].stream.read()
#         if b64 is not None:
#             s = HoughService()
#             s.load_from_bytes(b64)
#             circle = s.predict_retry()
#             print(f"{datetime.now()}: {circle}")
#             if circle is None:
#                 return jsonify(None)
#             return jsonify([int(circle[0]), int(circle[1]), int(circle[2]), float(circle[3])])
#     return "Bad image", 404

if __name__ == '__main__':
    app.run(host="0.0.0.0")

