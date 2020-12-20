import argparse
import time
from concurrent import futures

import cv2
import grpc
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from tqdm import tqdm

MODEL_NAME = "ssd"
SIGNATURE_NAME = "serving_default"
INPUT_NAMES = "inputs"
OUTPUT_NAMES = [
    "detection_boxes",
    "detection_classes",
    "detection_scores",
    "num_detections",
]
GRPC_OPTIONS = [("grpc.max_send_message_length", 1024 ** 3)]


def main(args):
    url = f"{args.url}:{args.port}"
    inputs = read_image(args.path)
    num = args.requests

    t = time.time()
    if args.ssl:
        infer_with_ssl(url, inputs, num, args.threads)
    else:
        infer_without_ssl(url, inputs, num, args.threads)
    dt = time.time() - t
    print(f"throughput: {num/dt:.2f} image/sec, latency: {dt:.2f} sec")


def infer_with_ssl(url, inputs, num, threads):
    cred = grpc.ssl_channel_credentials()
    with grpc.secure_channel(url, cred, options=GRPC_OPTIONS) as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        predict(stub, inputs, num, threads)


def infer_without_ssl(url, inputs, num, threads):
    with grpc.insecure_channel(url, options=GRPC_OPTIONS) as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        predict(stub, inputs, num, threads)


def predict(stub, inputs, num, threads):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    request.model_spec.signature_name = SIGNATURE_NAME
    request.inputs["inputs"].CopyFrom(tf.make_tensor_proto(inputs))

    with tqdm(total=num) as progress:
        with futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for _ in range(num):
                future = executor.submit(stub.Predict, request)
                future.add_done_callback(lambda x: progress.update())


def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image[np.newaxis, :, :, :]


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
    parser.add_argument("--requests", help="num of requests", type=int, default=100)
    parser.add_argument("--threads", help="num of threads", type=int, default=5)

    args = parser.parse_args()
    main(args)
