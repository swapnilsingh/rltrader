#!/bin/bash
set -e

NAMESPACE=rltrader
MANIFEST_DIR=k3s

if [[ "$1" == "start" ]]; then
  echo "📦 Ensuring namespace '$NAMESPACE' exists..."
  kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

  echo '🚀 Deploying Redis'
  kubectl apply -f $MANIFEST_DIR/redis.yaml -n $NAMESPACE
  echo "🟡 Redis deployment submitted — check with: kubectl get pods -n $NAMESPACE -l app=redis"

  echo '🚀 Deploying Inference'
  kubectl apply -f $MANIFEST_DIR/inference.yaml -n $NAMESPACE
  echo "🟡 Inference deployment submitted — check with: kubectl get pods -n $NAMESPACE -l app=inference"

  echo '🚀 Deploying Trainer'
  kubectl apply -f $MANIFEST_DIR/trainer.yaml -n $NAMESPACE
  echo "🟡 Trainer deployment submitted — check with: kubectl get pods -n $NAMESPACE -l app=trainer"

  if [[ -f $MANIFEST_DIR/ui.yaml ]]; then
    echo '🚀 Deploying UI'
    kubectl apply -f $MANIFEST_DIR/ui.yaml -n $NAMESPACE
    echo "🟡 UI deployment submitted — check with: kubectl get pods -n $NAMESPACE -l app=ui"
  else
    echo "⚠️ Skipping UI deployment (file not found)"
  fi

  echo "✅ All RLTrader components submitted to '$NAMESPACE'!"

elif [[ "$1" == "stop" ]]; then
  echo "🛑 Stopping RLTrader deployments in namespace '$NAMESPACE'..."
  kubectl delete -f $MANIFEST_DIR/ui.yaml -n $NAMESPACE --ignore-not-found
  kubectl delete -f $MANIFEST_DIR/trainer.yaml -n $NAMESPACE --ignore-not-found
  kubectl delete -f $MANIFEST_DIR/inference.yaml -n $NAMESPACE --ignore-not-found
  kubectl delete -f $MANIFEST_DIR/redis.yaml -n $NAMESPACE --ignore-not-found

  echo "✅ All RLTrader deployments stopped. Namespace '$NAMESPACE' retained."

else
  echo "❗ Usage: $0 {start|stop}"
  exit 1
fi
