# MLOps & Infrastructure for Machine Learning: Assignment 3
## Deploying PyTorch ML Workloads with Docker & Kubernetes

**Name:** Amritanshu Vivek  
**Roll No.:** DA25M545  
**Submission Date:** August 29, 2026  

---

## 1. Project Links
* **Public GitHub Repository:** [https://github.com/amritanshuvivek/mlops-pytorch-pipeline](https://github.com/amritanshuvivek/mlops-pytorch-pipeline)
* **Final Release Pull Request:** [https://github.com/amritanshuvivek/mlops-pytorch-pipeline/pull/1](https://github.com/amritanshuvivek/mlops-pytorch-pipeline/pull/1)

---

## 2. Project Architecture Diagram

Below is the end-to-end architecture of the MLOps pipeline, illustrating how configuration, datasets, container builds, and Kubernetes orchestration interact to execute training jobs and host model endpoints:

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

## 3. Reflection Write-Up

### Reflecting on MLOps Workload Deployment: Challenges and Key Learnings

Implementing the end-to-end MLOps pipeline for PyTorch workloads using Docker and Kubernetes presented a valuable learning curve, shifting my perspective from pure model development to robust infrastructure engineering. The assignment highlighted several real-world operational challenges, the most significant of which were permission management in container security and environment-specific folder overrides.

The most challenging part of the assignment was debugging the model serving Docker container within the Kubernetes cluster. In accordance with security best practices, the serving container was configured to run as a non-root user (`USER appuser`). However, because the dependencies were originally installed in the builder stage's `/root/.local` directory and copied directly to the runner stage, the system raised a `Permission Denied` error when trying to invoke Uvicorn. Resolving this required restructuring the multi-stage build. I utilized a prefix-based installation directory (`--prefix=/install`) in the compilation stage and copied these assets globally to `/usr/local` in the final runner stage. This kept the image lightweight, conformed to the security profile, and resolved the permission blocks.

Another operational hurdle was dataset retrieval during the training phase. The official torchvision downloader pulls the CIFAR-10 archive from the University of Toronto academic servers, which suffered from severe bandwidth limitations (averaging ~25 kB/s locally). Waiting hours for the download threatened to stall local container and cluster testing. To handle this, I engineered a `--smoke-test` execution flag. When toggled, the script bypasses torchvision downloads and instantiates a mock CIFAR-10 representation using `torchvision.datasets.FakeData`. This enabled instantaneous, deterministic training cycles (2 epochs in 12 seconds), allowing rapid end-to-end verification of the Kubernetes storage mounts and Job states.

Finally, managing the networking layer of the local Kubernetes cluster required careful orchestration. Since `kind` runs control-plane components inside Docker, host-level images are not automatically accessible. I had to explicitly load images using `kind load docker-image` commands and apply a structured version tagging strategy (`v2`) to prevent Kubelet from pulling stale cached layers.

In conclusion, this project demonstrated that deploying machine learning models involves much more than optimizing accuracy. Designing resilient container configurations, ensuring secure non-root directory ownership, and creating automated testing fallbacks are essential steps in bringing ML models to production securely and reliably.

---

## 4. Assignment Requirements Checklist

| Requirement | Status | Evidence / Document Reference |
| :--- | :--- | :--- |
| **Repository Structure** | **PASS** | File layout conforms to MLOps repository standards. |
| **Git workflow** | **PASS** | Main branch holds commits merged from develop/feature branches. |
| **PyTorch Model** | **PASS** | Modified ResNet-18 designed for CIFAR-10 in `src/model.py`. |
| **CIFAR-10 Dataset** | **PASS** | Normalization, augmentations, and data loaders in `src/dataset.py`. |
| **Config-driven training** | **PASS** | Read parameters from `configs/training_config.yaml`. |
| **JSON training logs** | **PASS** | Training progress logged as JSON lines to stdout. |
| **Early stopping** | **PASS** | Checks validation loss and triggers early stopping. |
| **Checkpoint** | **PASS** | Saves state dicts, epoch counts, and metrics to `classifier_v1.pt`. |
| **Training Dockerfile** | **PASS** | Built multi-stage slim container (`mlops-train:v2`). |
| **Serving Dockerfile** | **PASS** | Serving container built and runs as non-root `appuser`. |
| **Health endpoint** | **PASS** | `/health` returns 200 once weights are successfully loaded. |
| **Prediction endpoint** | **PASS** | `/predict` processes form file uploads and returns class probs. |
| **Kubernetes namespace** | **PASS** | All resources deploy in isolated `ml-training` namespace. |
| **ConfigMap** | **PASS** | Mounts configs dynamically inside training pods. |
| **Persistent storage** | **PASS** | PV/PVC claims allocate storage for datasets and weights. |
| **Kubernetes Job** | **PASS** | Training job completed successfully, writing checkpoint to PVC. |
| **Kubernetes Deployment**| **PASS** | Deployed model server with 2 healthy replicas. |
| **Liveness & Readiness Probes**| **PASS** | Health checks configured and passing on port 8080. |
| **Kubernetes Service** | **PASS** | ClusterIP service exposing port 80 targeting 8080. |
| **HPA** | **PASS** | Configured scale policies (2-5 replicas) targeting 80% CPU. |
| **End-to-end validation**| **PASS** | Successfully verified pipeline locally, in Docker, and in Kubernetes. |
| **Tests** | **PASS** | Pytest verified model layers, shapes, and exceptions. |
| **CI** | **PASS** | GitHub Action defined in `.github/workflows/ci.yml`. |

---

## 5. Verification Outputs Appendix

### Appendix A: Pytest Output
```text
============================= test session starts ==============================
platform darwin -- Python 3.10.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/amritanshuvivek/Documents/AI_Projects/69_docker_kubernetes/mlops-pytorch-pipeline
plugins: anyio-4.9.0, langsmith-0.4.10
collected 3 items

tests/test_model.py ...                                                  [100%]

============================== 3 passed in 1.86s ===============================
```

### Appendix B: Docker Build Training Output
```text
$ docker build -f docker/Dockerfile.train -t mlops-train:v2 .
#0 building with "desktop-linux" instance using docker driver
#1 [internal] load build definition from Dockerfile.train
...
#9 [builder 5/5] RUN pip install --no-cache-dir --prefix=/install -r train.txt
#9 Successfully installed MarkupSafe-3.0.3 filelock-3.32.4 fsspec-2026.7.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.6.1 numpy-2.4.6 pillow-12.3.0 pyyaml-6.0.2 sympy-1.14.0 torch-2.8.0 torchvision-0.23.0 typing-extensions-4.16.0
#10 [runner 3/5] COPY --from=builder /install /usr/local
#13 exporting to image
#13 naming to docker.io/library/mlops-train:v2 done
```

### Appendix C: Local Docker Training Run Output (Smoke Test)
```text
$ docker run --rm -e SMOKE_TEST=true -v $(pwd)/data:/app/data -v $(pwd)/checkpoints:/app/checkpoints mlops-train:v2
{"event": "training_init", "device": "cpu", "config": {"model": {"architecture": "resnet18", "num_classes": 10}, "training": {"epochs": 2, "batch_size": 64, "learning_rate": 0.001, "early_stopping_patience": 3}, "data": {"dataset": "cifar10", "data_dir": "/app/data"}, "output": {"checkpoint_dir": "/app/checkpoints", "model_name": "classifier_v1.pt"}}}
{"event": "smoke_test_mode", "message": "Using mock data for fast verification"}
{"event": "training_start", "total_epochs": 2}
{"epoch": 1, "train_loss": 2.438, "train_accuracy": 0.0898, "val_loss": 4.9582, "val_accuracy": 0.1562}
{"event": "checkpoint_saved", "epoch": 1, "val_loss": 4.9582, "val_accuracy": 0.1562, "path": "checkpoints/classifier_v1.pt"}
{"epoch": 2, "train_loss": 0.3502, "train_accuracy": 0.9961, "val_loss": 1.8918, "val_accuracy": 0.2969}
{"event": "checkpoint_saved", "epoch": 2, "val_loss": 1.8918, "val_accuracy": 0.2969, "path": "checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 1.8918, "best_val_accuracy": 0.2969, "checkpoint_path": "checkpoints/classifier_v1.pt"}
```

### Appendix D: Local Docker Serving Validation (curl)
```text
$ curl -i http://localhost:8080/health
HTTP/1.1 200 OK
server: uvicorn
content-length: 20
content-type: application/json

{"status":"healthy"}

$ curl -i -X POST http://localhost:8080/predict -F "image=@test_image.png"
HTTP/1.1 200 OK
server: uvicorn
content-length: 152
content-type: application/json

{"predicted_class":3,"predicted_label":"cat","probabilities":[0.023033,0.08905,0.213712,0.238022,0.087249,0.038015,0.153324,0.008948,0.042537,0.106109]}
```

### Appendix E: Kubernetes Training Job Run Status
```text
$ kubectl get job -n ml-training
NAME                 STATUS     COMPLETIONS   DURATION   AGE
mlops-training-job   Complete   1/1           15s        70s

$ kubectl logs -n ml-training -l job-name=mlops-training-job
{"event": "training_init", "device": "cpu", "config": {"model": {"architecture": "resnet18", "num_classes": 10}, "training": {"epochs": 2, "batch_size": 64, "learning_rate": 0.001, "early_stopping_patience": 3}, "data": {"dataset": "cifar10", "data_dir": "/app/data"}, "output": {"checkpoint_dir": "/app/checkpoints", "model_name": "classifier_v1.pt"}}}
{"event": "smoke_test_mode", "message": "Using mock data for fast verification"}
{"event": "training_start", "total_epochs": 2}
{"epoch": 1, "train_loss": 2.4563, "train_accuracy": 0.0977, "val_loss": 3.4938, "val_accuracy": 0.1406}
{"event": "checkpoint_saved", "epoch": 1, "val_loss": 3.4938, "val_accuracy": 0.1406, "path": "/app/checkpoints/classifier_v1.pt"}
{"epoch": 2, "train_loss": 0.3548, "train_accuracy": 0.9922, "val_loss": 2.0528, "val_accuracy": 0.2812}
{"event": "checkpoint_saved", "epoch": 2, "val_loss": 2.0528, "val_accuracy": 0.2812, "path": "/app/checkpoints/classifier_v1.pt"}
{"event": "training_complete", "best_val_loss": 2.0528, "best_val_accuracy": 0.2812, "checkpoint_path": "/app/checkpoints/classifier_v1.pt"}
```

### Appendix F: Kubernetes Model Serving Validation
```text
$ kubectl get deployment -n ml-training
NAME            READY   UP-TO-DATE   AVAILABLE   AGE
model-serving   2/2     2            2           98s

$ kubectl get svc -n ml-training
NAME            TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)   AGE
model-serving   ClusterIP   10.96.85.92   <none>        80/TCP    4m26s

# Port-forwarding tunnel opened: kubectl port-forward svc/model-serving 8080:80 -n ml-training

$ curl -i http://localhost:8080/health
HTTP/1.1 200 OK
server: uvicorn
content-length: 20
content-type: application/json

{"status":"healthy"}

$ curl -i -X POST http://localhost:8080/predict -F "image=@test_image.png"
HTTP/1.1 200 OK
server: uvicorn
content-length: 153
content-type: application/json

{"predicted_class":8,"predicted_label":"ship","probabilities":[0.081162,0.147572,0.09165,0.086626,0.111534,0.078215,0.101528,0.055336,0.163632,0.082745]}
```
<!-- End of Report -->
