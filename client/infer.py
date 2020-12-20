import argparse
import json
import time

import cv2
import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

MODEL_NAME = "ssd"
SIGNATURE_NAME = "serving_default"
INPUT_NAMES = "inputs"
OUTPUT_NAMES = [
    "detection_boxes",
    "detection_classes",
    "detection_scores",
    "num_detections",
]
GRPC_OPTIONS = []


def main(args):
    target = f"{args.url}:{args.port}"
    inputs = read_image(args.path)
    request = make_request(inputs)
    class_names = load_class_names()

    # inference
    t = time.time()
    if args.ssl:
        outputs = infer_with_ssl(target, request)
    else:
        outputs = infer_without_ssl(target, request)
    print(f"inference time: {(time.time() - t)*1000:.2f} ms")

    # print the result
    num = int(outputs["num_detections"][0])
    for i in range(num):
        detection_class_id = int(outputs["detection_classes"][0][i])
        detection_class = class_names[detection_class_id]
        detection_score = outputs["detection_scores"][0][i]
        print(f"detection {i + 1}: {detection_class}, {detection_score:.4f}")

    # drawing the result
    if args.out:
        image = cv2.cvtColor(inputs[0], cv2.COLOR_RGB2BGR)
        for i in range(num):
            xmin, xmax, ymin, ymax = outputs["detection_boxes"][0][i]
            image = draw_bounding_box_on_image(
                image,
                xmin,
                xmax,
                ymin,
                ymax,
                label=detection_class,
                score=detection_score,
            )
        cv2.imwrite(args.out, image)


def make_request(inputs: np.ndarray) -> predict_pb2.PredictRequest:
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    request.model_spec.signature_name = SIGNATURE_NAME
    request.inputs["inputs"].CopyFrom(tf.make_tensor_proto(inputs))
    return request


def infer_with_ssl(target, request):
    cred = grpc.ssl_channel_credentials()
    with grpc.secure_channel(target, cred, options=GRPC_OPTIONS) as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        outputs = predict(stub, request)
    return outputs


def infer_without_ssl(target, request):
    with grpc.insecure_channel(target, options=GRPC_OPTIONS) as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        outputs = predict(stub, request)
    return outputs


def predict(stub, request):
    result = stub.Predict(request)
    return {name: tf.make_ndarray(result.outputs[name]) for name in OUTPUT_NAMES}


def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image[np.newaxis, :, :, :]


def load_class_names():
    with open("mscoco_label_map.json") as fp:
        data = json.load(fp)
    class_names = {row["id"]: row["display_name"] for row in data}
    return class_names


def draw_bounding_box_on_image(
    image,
    ymin,
    xmin,
    ymax,
    xmax,
    color=(0, 0, 255),
    thickness=2,
    label="",
    score="",
    use_normalized_coordinates=True,
):
    image = np.copy(image)

    if use_normalized_coordinates:
        im_height, im_width = image.shape[:2]
        (left, right, top, bottom) = (
            xmin * im_width,
            xmax * im_width,
            ymin * im_height,
            ymax * im_height,
        )
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    p1 = (int(left), int(top))
    p2 = (int(right), int(bottom))

    image = cv2.rectangle(image, p1, p2, color, thickness)

    if label and score:
        p = (int(left) + 5, int(top) + 20)
        text = f"{label} ({score * 100:.1f}%)"
        image = cv2.putText(
            image,
            text,
            p,
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            color,
            lineType=cv2.LINE_AA,
        )

    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="the path of input image")
    parser.add_argument(
        "--url",
        help="the URL of tensorflow serving (default: localhost)",
        default="localhost",
    )
    parser.add_argument(
        "--port", help="the port of tensorflow serving (default: 8500)", default="8500"
    )
    parser.add_argument("--ssl", help="use SSL connection", action="store_true")
    parser.add_argument("-o", "--out", help="the path of output image")

    args = parser.parse_args()
    main(args)
