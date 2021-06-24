# TensorFlow Serving + Application Load Balancer Example

This repository is an example for inference with [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) via [AWS Application Load Balancer](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/introduction.html) using gRPC.

Note: [New – Application Load Balancer Support for End-to-End HTTP/2 and gRPC](https://aws.amazon.com/jp/blogs/aws/new-application-load-balancer-support-for-end-to-end-http-2-and-grpc/)

## Requirements

- Docker
- AWS
  - VPC
  - subnets (public / private)
- awscli
- ecs-cli

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

### Run with ECS

```sh
ecs-cli configure --cluster tfserving-alb-tutorial --default-launch-type EC2 --region ap-northeast-1
```

## Step 2. Request from client
