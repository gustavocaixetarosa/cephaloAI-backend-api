from flask import Flask, Response, request, jsonify
from flask_cors import cross_origin, CORS
import os

from model import fusionVGG19, dilationInceptionModule
from imagem_service import ImagemService
import angle

service = ImagemService("models/proccess_dataTeste.pkl")

app = Flask(__name__)
CORS(app)


@app.route("/processar-imagem", methods=["POST"])
@cross_origin()
def processar() -> Response:
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]
    img_temp_path = f"temp_{file.filename}"
    file.save(img_temp_path)

    try:
        outputs = service.predict(img_temp_path)

        coords, _, _ = service.model.getCoordinate(outputs)
        print("printing coords")
        print(coords)
        coords_list = coords.squeeze(0).cpu().numpy().tolist()
        print("printing coords_list")
        print(coords_list)

        points = [angle.Point(x, y) for x, y in coords_list]

        angles = angle.classification(points)
        print(angles)

        return jsonify({"coords": coords_list, "angles": angles})

    finally:
        if os.path.exists(img_temp_path):
            os.remove(img_temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
