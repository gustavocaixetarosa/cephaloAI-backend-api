from flask import Flask, Response, request, jsonify
import os

from imagem_service import ImagemService

service = ImagemService("models/proccess_dataTeste.pkl")

app = Flask(__name__)


@app.route("/processarimagem", methods=["POST"])
def processar() -> Response:
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files["file"]
    img_temp_path = f"temp_{file.filename}"
    file.save(img_temp_path)

    try:
        outputs = service.predict(img_temp_path)

        coords, _, _ = service.model.getCoordinate(outputs)
        coords_list = coords.squeeze(0).cpu().numpy().tolist()

        return jsonify({"landkmarkds": coords_list})
    finally:
        if os.path.exists(img_temp_path):
            os.remove(img_temp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
