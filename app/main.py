from flask import Flask, request, jsonify

from torch_utils import transform_image, get_prediction

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('files')
        if file is None or file.filename == "":
            return jsonify({"error": "format not supported"})
        if not allowed_file(file.filename):
            return jsonify({"error": "format not supported"})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            get_pred = get_prediction(tensor)
            pred = jsonify({"prediction": get_pred.item(), "className": str(get_pred.item())})
            return jsonify(pred)

        except:
            return jsonify({"error": "error during prediction"})
