# TensorFlow Serving + Application Load Balancer Example

This repository is an example for inference with [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) via [AWS Application Load Balancer](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html) using gRPC.

Note: [New – Application Load Balancer Support for End-to-End HTTP/2 and gRPC](https://aws.amazon.com/jp/blogs/aws/new-application-load-balancer-support-for-end-to-end-http-2-and-grpc/)

## Requirements

- Docker
- awscli
- Python 3.8

## AWS Architecture

![tfserving+alb-min](https://user-images.githubusercontent.com/6437204/102708909-4f4a1380-42e9-11eb-80e9-7d05fdd5eadc.png)

## Step 1. Create inference server

In this step, the working direcotry is `server/cpu`.

```sh
cd server/cpu
```

### Build docker image

```sh
docker build serving-cpu .
```

### Run locally

```sh
docker run -d -p 8500:8500 serving-cpu
```

### Run with ECS (without ALB)

Upload the docker image to ECR.

```sh
docker tag serving-cpu 111111111111.dkr.ecr.ap-northeast-1.amazonaws.com/serving-cpu
aws ecr get-login-password --region ap-northeast-1 \
 | docker login --username AWS --password-stdin 111111111111.dkr.ecr.ap-northeast-1.amazonaws.com
docker push 111111111111.dkr.ecr.ap-northeast-1.amazonaws.com/serving-cpu
```

Register the task definition.

```sh
cp ecs/task.json{.example,}
aws ecs register-task-definition --cli-input-json file://ecs/task.json
```

Create the service.

```sh
cp ecs/service-without-alb.json{.example,}
aws ecs create-service --cli-input-json file://ecs/service-without-alb.json
```

### Run with ECS (with ALB)

Create the service.

```sh
cp ecs/service-with-alb.json{.example,}
aws ecs create-service --cli-input-json file://ecs/service-with-alb.json
```

### Run with ECS (with ALB, with AutoScaling)

To be written

## Step 2. Request from client

In this step, the working direcotry is `client`.

```sh
cd client
```

Install python pacages using [poetry](https://python-poetry.org/).

```sh
poetry install
```

### Infer

```sh
python infer.py /path/to/input.jpg -o /path/to/output.jpg
```

If you want to use a specific server and SSL, do the following:

```sh
python infer.py --url example.com --port 443 --ssl /path/to/input.jpg
```
