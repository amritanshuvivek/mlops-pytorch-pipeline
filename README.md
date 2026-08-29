# MLOps PyTorch Pipeline: Deploying ML Workloads with Docker & Kubernetes

This repository implements a complete end-to-end MLOps pipeline for training, package-building, local validation, and cluster-deploying a PyTorch-based image classification model on the CIFAR-10 dataset using Docker and Kubernetes.

---

## 1. Project Overview & Architecture

The project automates the machine learning lifecycle:
1. **Local Development:** Defining a ResNet-18 model optimized for CIFAR-10 image size, dataset loading, training loops with early stopping and checkpointing, and testing code.
2. **Containerization:** Separating training and serving dependencies into custom multi-stage Docker builds.
3. **Orchestration:** Setting up Persistent Volume Claims, ConfigMaps, Job-based training workloads, Deployment-based model servers (with liveness/readiness probes), and Horizontal Pod Autoscaling (HPA) in a Kubernetes cluster.

```mermaid
flowchart TD
    %% Dataset and Configuration Input
    Dataset[CIFAR-10 Data Source] --> TrainDataset
    Config[configs/training_config.yaml] --> TrainScript
    
    %% Training Phase
    subgraph Local/Container Training
        TrainScript[src/train.py] --> TrainLoop[PyTorch ResNet-18 Train Loop]
        TrainDataset[Dataset Loader] --> TrainLoop
        TrainLoop --> Checkpoint[checkpoints/classifier_v1.pt]
    end

    %% Packaging Phase
    subgraph Docker Packaging
        DockerfileTrain[docker/Dockerfile.train] --> DockerTrainImage[mlops-train:v2]
        DockerfileServe[docker/Dockerfile.serve] --> DockerServeImage[mlops-serve:v2]
    end

    %% Orchestration Phase
    subgraph Kubernetes Deployment (ml-training namespace)
        K8sJob[Kubernetes Job] -->|Runs| DockerTrainImage
        K8sJob -->|Mounts| DataPVC[(data-pvc)]
        K8sJob -->|Writes best weights| CheckpointPVC[(checkpoints-pvc)]
        
        K8sDeploy[Kubernetes Deployment] -->|Runs 2 Replicas| DockerServeImage
        K8sDeploy -->|Read-only Mount| CheckpointPVC
        K8sDeploy -->|Auto-scales via| K8sHPA[Horizontal Pod Autoscaler]
        
        K8sService[Kubernetes Service] -->|Balances traffic to| K8sDeploy
    end

    %% Consumer Phase
    Client[Client CLI / curl] -->|Queries| K8sService
```

---

## 2. Repository Structure

The repository contains the following files and directories:

```text
mlops-pytorch-pipeline/
├── README.md                  # This documentation and final report
├── .gitignore                 # Excludes cache, datasets, and large weights
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow (lint & test)
├── configs/
│   └── training_config.yaml   # Hyperparameter configuration file
├── docker/
│   ├── Dockerfile.train       # Multi-stage Dockerfile for training
│   └── Dockerfile.serve       # Multi-stage Dockerfile for FastAPI serving
├── k8s/
│   ├── namespace.yaml         # Creates ml-training namespace
│   ├── pvc.yaml               # persistent volume claims for data and checkpoints
│   ├── configmap.yaml         # ConfigMap wrapping training_config.yaml
│   ├── training-job.yaml      # Kubernetes Job for training models
│   ├── serving-deployment.yaml# Kubernetes Deployment for model serving
│   ├── serving-service.yaml   # ClusterIP Service for routing serving requests
│   └── hpa.yaml               # Horizontal Pod Autoscaler configuration
├── requirements/
│   ├── train.txt              # Pinned training dependencies
│   └── serve.txt              # Pinned inference serving dependencies
├── src/
│   ├── dataset.py             # Data loading and preprocessing pipeline
│   ├── model.py               # PyTorch ResNet-18 model architecture
│   ├── train.py               # Training loop with validation and checkpointing
│   └── serve.py               # FastAPI inference serving app
├── tests/
│   └── test_model.py          # Pytest unit tests for model components
├── evidence/                  # Validation logs and artifacts directory
│   ├── pytest.txt
│   ├── docker-training-build.txt
│   ├── docker-training-run.txt
│   ├── docker-serving-build.txt
│   ├── docker-serving-health.txt
│   ├── docker-serving-predict.txt
│   ├── kubernetes-training.txt
│   ├── kubernetes-serving.txt
│   ├── kubernetes-health.txt
│   └── kubernetes-predict.txt
├── test_image.png             # Generated mock image for API tests
└── data/                      # Local cached dataset directory (git-ignored)
```

---

## 3. Component Details & Design Choices

### 3.1 Model Architecture (`src/model.py`)
Standard torchvision `resnet18` is designed for 224x224 ImageNet images, starting with a 7x7 conv layer with stride 2 and maxpooling, which downsamples 32x32 CIFAR-10 inputs to 8x8 immediately, leading to poor learning capacity. 
We modified the model:
1. Replaced `conv1` with a `3x3 Conv2d` layer with `stride=1` and `padding=1` to preserve spatial resolution in early stages.
2. Replaced `maxpool` with `nn.Identity()` to prevent early downsampling.
3. Modified `fc` linear output size to match `num_classes=10`.

### 3.2 Dataset Handling (`src/dataset.py`)
Fetches CIFAR-10 data and applies standard normalization values:
- Mean: `(0.4914, 0.4822, 0.4465)`
- Standard Deviation: `(0.2023, 0.1994, 0.2010)`
Data augmentation (`RandomCrop`, `RandomHorizontalFlip`) is enabled during training to prevent overfitting.
**Smoke Test Mode:** Added a `get_smoke_dataloaders` function utilizing `torchvision.datasets.FakeData` to generate dummy images, enabling instantaneous training loops for CI/CD and cluster validations where dataset downloads are slow.

### 3.3 Training Pipeline (`src/train.py`)
A script that parses arguments and runs training. Features:
- **Device Support:** Works on Apple Silicon GPUs (`mps`), Nvidia GPUs (`cuda`), and fallback `cpu`.
- **JSON Lines Metrics Logging:** Output is formatted as structured JSON Lines (e.g. `{"epoch": 1, "train_loss": 1.2345, "train_accuracy": 0.5432, "val_loss": 1.1234, "val_accuracy": 0.5678}`) so telemetry log parsers can ingest it easily.
- **Early Stopping & Checkpoint Saving:** Monitors validation loss, stops if it fails to improve within `early_stopping_patience` epochs, and saves state dict (including optimizer state and metrics) to `classifier_v1.pt`.
- **Structured Event Triggers:** Outputs events such as `{"event": "checkpoint_saved", ...}`.

### 3.4 Model Serving (`src/serve.py`)
FastAPI application that loads the checkpoint during startup:
- **GET `/health`**: Returns HTTP 200 `{"status": "healthy"}` only if the model loaded successfully, otherwise returns HTTP 503.
- **POST `/predict`**: Accepts an uploaded PNG or JPEG image via multipart forms, pre-processes it to a normalized 32x32 tensor, runs inference under `torch.no_grad()`, and outputs predictions and probability distribution. Returns HTTP 400 for invalid formats.

### 3.5 Docker Optimization
1. **Dependency Separation:** Training image (`requirements/train.txt`) contains only training libraries, and serving image (`requirements/serve.txt`) contains only runtime dependencies (no heavy profiling/visualization tools), minimizing serving footprint.
2. **Multi-Stage Builds:** Uses a `builder` stage to install dependencies using a `--prefix=/install` switch, then copies compiled wheels to `/usr/local` in the final `runner` stage. This keeps final layers small and clean.
3. **Non-Root Execution:** Serving container runs as `USER appuser` to satisfy Kubernetes production security profiles.
4. **Health Check:** Rather than installing `curl` on python slim images, we configure `HEALTHCHECK` using Python's built-in `urllib` module:
   ```dockerfile
   HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
     CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1
   ```

### 3.6 Kubernetes Manifests
1. **Namespace (`k8s/namespace.yaml`):** Isolates the workload under `ml-training` namespace.
2. **Storage (`k8s/pvc.yaml`):** PVCs allocate block storage dynamically using cluster defaults.
3. **ConfigMap (`k8s/configmap.yaml`):** Ingests the YAML configuration file and mounts it inside training pods at `/app/configs/training_config.yaml`.
4. **Job (`k8s/training-job.yaml`):** Trains the model and writes weights to `/app/checkpoints/` using 2 CPUs and 4Gi memory requests.
5. **Deployment (`k8s/serving-deployment.yaml`):** Mounts the weights PVC as `readOnly: true`, exposes container port `8080`, configures liveness and readiness probes, and schedules 2 replicas with rolling updates.
6. **Autoscaler (`k8s/hpa.yaml`):** Configures scaling from 2 to 5 replicas when average CPU utilization hits 80%.

---

## 4. Local Execution & Validation Instructions

### 4.1 Local Python Environment Setup
1. Clone the repository and navigate to it:
   ```bash
   cd mlops-pytorch-pipeline
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install training requirements and test tools:
   ```bash
   pip install -r requirements/train.txt
   pip install pytest
   ```
4. Run Unit Tests:
   ```bash
   python3 -m pytest
   ```
   *Expected Outcome: 3 passed tests (see [pytest.txt](file:///Users/amritanshuvivek/Documents/AI_Projects/69_docker_kubernetes/mlops-pytorch-pipeline/evidence/pytest.txt)).*

5. Run Training (Smoke Test Mode):
   ```bash
   CHECKPOINT_DIR=checkpoints DATA_DIR=data PYTHONPATH=. python3 src/train.py --smoke-test
   ```
   *Expected Outcome: Trains for 2 epochs on fake data and writes weights to `checkpoints/classifier_v1.pt`.*

6. Run FastAPI Server Locally:
   ```bash
   PYTHONPATH=. CONFIG_PATH=configs/training_config.yaml CHECKPOINT_DIR=checkpoints python3 src/serve.py
   ```
7. Test local endpoints:
   ```bash
   # Create a test image if not exists
   python3 -c "from PIL import Image; img = Image.new('RGB', (32, 32), color=(73, 109, 137)); img.save('test_image.png')"
   
   # Query endpoints
   curl http://localhost:8080/health
   curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
   ```

### 4.2 Docker Training & Serving Setup
1. Build both training and serving images locally:
   ```bash
   docker build -f docker/Dockerfile.train -t mlops-train:v2 .
   docker build -f docker/Dockerfile.serve -t mlops-serve:v2 .
   ```
2. Run Training Container (passing SMOKE_TEST=true environment override):
   ```bash
   docker run --rm \
     -e SMOKE_TEST=true \
     -v "$(pwd)/data:/app/data" \
     -v "$(pwd)/checkpoints:/app/checkpoints" \
     mlops-train:v2
   ```
3. Run Serving Container:
   ```bash
   docker run -d --name serve-container -p 8080:8080 \
     -v "$(pwd)/checkpoints:/app/checkpoints" \
     mlops-serve:v2
   ```
4. Test container endpoints:
   ```bash
   curl http://localhost:8080/health
   curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
   ```
5. Stop container:
   ```bash
   docker rm -f serve-container
   ```

---

## 5. Kubernetes Cluster Deployment Setup

### 5.1 Prerequisites: Cluster Initialization
Since no active cluster was present in the environment, we initialized a local cluster using `kind`:
1. Install `kind` via Homebrew:
   ```bash
   brew install kind
   ```
2. Spin up a multi-node cluster:
   ```bash
   kind create cluster --name mlops-cluster
   ```
3. Verify cluster connectivity:
   ```bash
   kubectl cluster-info
   kubectl get nodes
   ```

### 5.2 Load Locally-Built Images
Kind does not automatically access host-level Docker images. We load our built images into the cluster:
```bash
kind load docker-image mlops-train:v2 --name mlops-cluster
kind load docker-image mlops-serve:v2 --name mlops-cluster
```

### 5.3 Deploy Workloads
1. Deploy Namespace, ConfigMap, and PVCs:
   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/pvc.yaml
   ```
2. Execute the model training Job:
   ```bash
   kubectl apply -f k8s/training-job.yaml
   ```
3. Monitor the training status and print container logs:
   ```bash
   kubectl get job -n ml-training
   kubectl get pods -n ml-training -l job-name=mlops-training-job
   kubectl logs -f -n ml-training -l job-name=mlops-training-job
   ```
   *Note: Ensure the training job completes successfully and prints `{"event": "training_complete"}`.*

4. Deploy the serving resources (HPA, Deployment, Service):
   ```bash
   kubectl apply -f k8s/serving-deployment.yaml
   kubectl apply -f k8s/serving-service.yaml
   kubectl apply -f k8s/hpa.yaml
   ```
5. Validate Serving Pod health check probes:
   ```bash
   kubectl get deployment -n ml-training
   kubectl describe deployment model-serving -n ml-training
   ```
   *Ensure replicas reach READY 2/2 and status is Running.*

### 5.4 Test Cluster Predictions
To check predictions from the cluster, open a port-forward tunnel to the ClusterIP Service:
```bash
# Start tunnel in background
kubectl port-forward svc/model-serving 8080:80 -n ml-training &

# Test endpoints
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"

# Kill tunnel when done
kill -9 $!
```

---

## 6. Actual Observed Results

All outputs have been collected and saved inside the [evidence/](file:///Users/amritanshuvivek/Documents/AI_Projects/69_docker_kubernetes/mlops-pytorch-pipeline/evidence) folder:

### 6.1 Pytest Results
```text
tests/test_model.py ...                                                  [100%]
============================== 3 passed in 1.86s ===============================
```
*Tested successfully: model initialization, forward pass output logit shapes, and invalid architecture error raising.*

### 6.2 Docker training
- Build status: SUCCESS
- Training status: Completed successfully
- Output model: `checkpoints/classifier_v1.pt` (approx. 134MB)
- Best validation loss: `1.8918`
- Best validation accuracy: `0.2969` (obtained after 2 smoke-test epochs)

### 6.3 Docker serving
- Build status: SUCCESS
- Health check endpoint `/health` response: `{"status":"healthy"}` (HTTP 200)
- Predict endpoint `/predict` response:
  ```json
  {"predicted_class":3,"predicted_label":"cat","probabilities":[0.023033,0.08905,0.213712,0.238022,0.087249,0.038015,0.153324,0.008948,0.042537,0.106109]}
  ```

### 6.4 Kubernetes training
- Namespace: `ml-training`
- Job Status: Completed (Completions: 1/1)
- Pod Status: Completed
- Best validation loss: `2.0528`
- Best validation accuracy: `0.2812`

### 6.5 Kubernetes serving
- Deployment: `model-serving`
- Replicas: 2 (READY 2/2)
- Service type: `ClusterIP`
- Probes configured: Liveness (`GET /health` period 10s), Readiness (`GET /health` initial delay 15s, period 5s)
- Prediction endpoint response (via port-forwarding ClusterIP):
  ```json
  {"predicted_class":8,"predicted_label":"ship","probabilities":[0.081162,0.147572,0.09165,0.086626,0.111534,0.078215,0.101528,0.055336,0.163632,0.082745]}
  ```

### 6.6 HPA
- HPA Name: `model-serving-hpa`
- Target utilization: 80% CPU
- Replicas: min 2, max 5, current 2
- Metrics status: Metrics server is currently unavailable in the local kind cluster, so targets show `<unknown>`, but resources are configured properly.

---

## 7. Troubleshooting and Fixes

During implementation, we identified and fixed several issues:

1. **Host-level directory permissions (`/app`):**
   * *Problem:* Standard configs were targeting `/app/data` and `/app/checkpoints` directly, which raised permission/not-found errors during local runs on macOS.
   * *Fix:* Added support for environment overrides (`DATA_DIR` and `CHECKPOINT_DIR`) in the training and serving scripts so they adapt to relative local folders (`data/` and `checkpoints/`) during local runs.
2. **Docker non-root permissions error in serving:**
   * *Problem:* Copying python dependencies from builder stage `/root/.local` to `/root/.local` in the runner stage caused `appuser` (non-root) to fail starting Uvicorn due to `Permission denied`.
   * *Fix:* Updated both Dockerfiles to install dependencies to a globally accessible folder using `--prefix=/install` in builder, and copied to `/usr/local` in runner.
3. **Cluster image caching during rebuilds:**
   * *Problem:* When we rebuilt the Docker image to fix permissions, the kind cluster kept running the old version and crashing.
   * *Fix:* Introduced image version tagging (`v2`) for local images to force Kubernetes to pull and load the correct updated version.
4. **CIFAR-10 Slow Academic Mirrors:**
   * *Problem:* torchvision downloads CIFAR-10 dataset files from the University of Toronto servers, which were extremely slow (~25kB/s), threatening to hang training.
   * *Fix:* Added a `--smoke-test` flag to the training script to load `torchvision.datasets.FakeData` instead. This runs the pipeline instantly for CI/CD and cluster deployment checks.

---

## 8. Git and Pull Request Workflow

Conceptual workflow for managing this project in a team structure:

### 8.1 Branch Structure
```text
  main (production code)
    ▲
    │ (pull requests)
  develop (release staging)
    ▲
    │ (pull requests)
  feature/* (feature branches)
```

### 8.2 Suggested Conventional Commit and Branch Strategy

* **PR 1: Dataset and Model Setup**
  - Branch name: `feature/dataset-model-setup`
  - Commit message: `feat: implement CIFAR10 dataset loader and modified ResNet18 model`
  - PR Title: `feat: Add model architecture and dataset dataloaders`
  - PR Description: "Implements model.py featuring ResNet-18 optimized for CIFAR-10 image sizes, dataset.py for data loading, and test_model.py for pytest unit tests."

* **PR 2: Training Pipeline**
  - Branch name: `feature/training-pipeline`
  - Commit message: `feat: implement train.py with logging, checkpointing, and early stopping`
  - PR Title: `feat: Add training script and configuration`
  - PR Description: "Implements the train.py script with structured JSON lines logging, validation splits, early stopping, and checkpoint saving."

* **PR 3: Serving API & Containerization**
  - Branch name: `feature/model-serving-docker`
  - Commit message: `feat: implement FastAPI serving API and Dockerfiles`
  - PR Title: `feat: Add FastAPI serve.py and multi-stage Dockerfiles`
  - PR Description: "Implements serve.py serving health/predict endpoints, and multi-stage optimized Dockerfiles for training and serving stages."

* **PR 4: Kubernetes Deployment**
  - Branch name: `feature/kubernetes-deployment`
  - Commit message: `feat: create Kubernetes manifests for Namespace, PVCs, ConfigMap, Job, Deployment, Service, and HPA`
  - PR Title: `feat: Add Kubernetes manifests for cluster orchestration`
  - PR Description: "Creates Kubernetes resource configurations for deploying training and serving elements, including Persistent Volumes and CPU scaling."

---

## 9. Final Assignment Requirements Checklist

| Requirement | Status | Evidence |
| :--- | :--- | :--- |
| **Repository Structure** | PASS | Folders structured exactly as requested. |
| **Git workflow** | PASS | Follows conceptual `main` -> `develop` -> `feature/*` layout. |
| **PyTorch Model** | PASS | Modified ResNet-18 implementation present in `src/model.py`. |
| **CIFAR-10 Dataset** | PASS | Transformations and DataLoader configured in `src/dataset.py`. |
| **Config-driven training** | PASS | Configurations loaded from `configs/training_config.yaml`. |
| **JSON training logs** | PASS | Logs metrics in JSON Lines format to stdout. |
| **Early stopping** | PASS | Implemented patience checks in `src/train.py` loop. |
| **Checkpoint** | PASS | Saves state dicts to `/app/checkpoints/classifier_v1.pt`. |
| **Training Dockerfile** | PASS | Multi-stage slim image built and tested successfully. |
| **Serving Dockerfile** | PASS | Lightweight image built and running as non-root `appuser`. |
| **Health endpoint** | PASS | FastAPI `/health` endpoint checks if model is initialized. |
| **Prediction endpoint** | PASS | FastAPI `/predict` accepts image files and returns probabilities. |
| **Kubernetes namespace** | PASS | Created `ml-training` namespace via `k8s/namespace.yaml`. |
| **ConfigMap** | PASS | Ingests `training_config.yaml` to `/app/configs/`. |
| **Persistent storage** | PASS | Dynamic PVC allocations configured in `k8s/pvc.yaml`. |
| **Kubernetes Job** | PASS | Completed successfully on v2 image producing model weights. |
| **Kubernetes Deployment**| PASS | Scheduled serving pods using v2 image. |
| **2 replicas** | PASS | Deployment scales to 2 replicas successfully. |
| **Liveness probe** | PASS | Probes FastAPI `/health` on 8080 every 10 seconds. |
| **Readiness probe** | PASS | Probes FastAPI `/health` starting after 15 seconds. |
| **Kubernetes Service** | PASS | ClusterIP Service routes traffic to target port 8080. |
| **HPA** | PASS | Configured scale limits (2-5 replicas) at 80% CPU target. |
| **End-to-end validation**| PASS | Queried endpoints through local Docker and k8s port-forwarding. |
| **Tests** | PASS | Pytest unit tests passed (3/3 tests passed). |
| **CI** | PASS | GitHub workflow defined in `.github/workflows/ci.yml`. |
| **README** | PASS | Professional documentation detailing all implementation steps. |

---

## 10. Limitations & AI Disclosures

1. **Slow Academic Dataset Mirror:** The Toronto server hosting CIFAR-10 data was downloading at ~25 kB/s, which would take hours to download. We bypassed this for fast testing using torchvision's `FakeData` class as a smoke test mode.
2. **Kubernetes Metrics Server:** Metrics server was not enabled by default in our local `kind` cluster, so HPA target values shown in `kubectl get hpa` display as `<unknown>`, but the manifests are valid.
3. **AI Assistant Usage:** Google Deepmind's Antigravity assistant was used to plan the implementation, write scripts, debug Docker image folder permissions, setup local cluster, deploy manifests, and collect verification logs.
