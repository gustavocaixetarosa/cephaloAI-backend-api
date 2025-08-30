from flask import Flask, Response, request, jsonify, send_file
from flask_cors import cross_origin, CORS
import os

from model import fusionVGG19, dilationInceptionModule
from imagem_service import ImagemService, desenhar_pontos
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
        print("Printing angles")
        print(angles)

        img_overlay_path = f"ovl_{file.filename}"
        desenhar_pontos(img_temp_path, coords_list, img_overlay_path)
        print("Imagem com overlay salva em: " + img_overlay_path)
        return jsonify(
            {
                "coords": coords_list,
                "angles": angles,
                "image_with_overlay_path": img_overlay_path,
            }
        )

    finally:
        if os.path.exists(img_temp_path):
            os.remove(img_temp_path)


@app.route("/download-imagem/<filename>")
def download_imagem(filename):
    return send_file(filename, mimetype="image/png", as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
