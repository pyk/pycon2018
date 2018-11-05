from typing import Any

from flask import Flask
from flask import request
from flask import jsonify

import torch

from model import IrisClassifier


app = Flask(__name__)

# Initialize the model
model = IrisClassifier()


@app.route("/")  # type: ignore
def index() -> Any:
    return "PyCon example!"


@app.route("/predict")  # type: ignore
def predict() -> Any:
    sepal_length = int(request.args.get("sepal_length", 0))
    sepal_width = int(request.args.get("sepal_width", 0))
    petal_length = int(request.args.get("petal_length", 0))
    petal_width = int(request.args.get("petal_width", 0))
    # TODO(pyk): perform validation

    # TODO: build the input
    x = torch.Tensor([sepal_length, sepal_width, petal_length, petal_width])
    prediction = model.predict(x)

    return jsonify(
        {
            "Setosa": prediction[0],
            "Versicolour": prediction[1],
            "Virginica": prediction[2],
        }
    )


def main() -> None:
    # Load the trained file
    model_path = "IrisClassifier.pickle"
    model.load_state_dict(torch.load(model_path))

    app.run(port=9999, debug=True)


if __name__ == "__main__":
    main()
