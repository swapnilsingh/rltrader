#!/bin/bash
set -e

REGISTRY="192.168.1.200:30000"
TAG="latest"
MODULES=("inference" "ui")

build_and_push() {
  local name=$1
  local dockerfile=$2

  echo "🔨 Building $name..."
  docker build -t $REGISTRY/rl-trader-$name:$TAG -f $dockerfile .

  echo "📤 Pushing $name to registry..."
  docker push $REGISTRY/rl-trader-$name:$TAG
}

for module in "${MODULES[@]}"; do
  build_and_push "$module" "src/$module/Dockerfile"
done

# Special case: Trainer with GPU detection
echo "🧠 Detecting Jetson environment for trainer build..."
if grep -q "tegra" /proc/device-tree/model 2>/dev/null || uname -a | grep -qi "tegra"; then
  echo "🚀 Jetson detected. Building GPU-enabled trainer image..."
  docker build \
    --platform linux/arm64/v8 \
    --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r35.3.1-pth2.0-py3 \
    -t $REGISTRY/rl-trader-trainer:$TAG \
    -f src/trainer/Dockerfile.gpu \
    .
  docker push $REGISTRY/rl-trader-trainer:$TAG
else
  echo "⚠️ Non-GPU system detected. Building trainer with CPU fallback..."
  build_and_push "trainer" "src/trainer/Dockerfile"
fi

echo "✅ All images built and pushed successfully!"