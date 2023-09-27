Install minikube on local system using the below command

```
brew install minikube
```

If you have docker on local system then you can use Kubernetes using docker

```
minikube start --driver=docker
```

1.kubectl describe <your_deployment>

```
Name:                   clip-classifer-deployment
Namespace:              default
CreationTimestamp:      Tue, 26 Sep 2023 17:57:15 +0530
Labels:                 app=clip-classifier
Annotations:            deployment.kubernetes.io/revision: 1
Selector:               app=clip-classifier
Replicas:               2 desired | 2 updated | 2 total | 2 available | 0 unavailable
StrategyType:           RollingUpdate
MinReadySeconds:        0
RollingUpdateStrategy:  25% max unavailable, 25% max surge
Pod Template:
  Labels:  app=clip-classifier
  Containers:
   clip-classifier:
    Image:        clip-k8s:latest
    Port:         7860/TCP
    Host Port:    0/TCP
    Environment:  <none>
    Mounts:       <none>
  Volumes:        <none>
Conditions:
  Type           Status  Reason
  ----           ------  ------
  Progressing    True    NewReplicaSetAvailable
  Available      True    MinimumReplicasAvailable
OldReplicaSets:  <none>
NewReplicaSet:   clip-classifer-deployment-5cc8894f98 (2/2 replicas created)
Events:
  Type    Reason             Age   From                   Message
  ----    ------             ----  ----                   -------
  Normal  ScalingReplicaSet  7m    deployment-controller  Scaled up replica set clip-classifer-deployment-5cc8894f98 to 2
```

2. kubectl describe <your_pod>

```
Name:             clip-classifer-deployment-5cc8894f98-lc4v2
Namespace:        default
Priority:         0
Service Account:  default
Node:             minikube/192.168.49.2
Start Time:       Tue, 26 Sep 2023 17:57:15 +0530
Labels:           app=clip-classifier
                  pod-template-hash=5cc8894f98
Annotations:      <none>
Status:           Running
IP:               10.244.0.37
IPs:
  IP:           10.244.0.37
Controlled By:  ReplicaSet/clip-classifer-deployment-5cc8894f98
Containers:
  clip-classifier:
    Container ID:   docker://5bb743c319ca45fd6efd40764c427f0c94467f71906c932e1768d50ef1646fb3
    Image:          clip-k8s:latest
    Image ID:       docker://sha256:5a24132d49ba699f87e698f124352e401aa4ae04e11a7cb3ea02e91a9f63a17e
    Port:           7860/TCP
    Host Port:      0/TCP
    State:          Running
      Started:      Tue, 26 Sep 2023 18:03:33 +0530
    Last State:     Terminated
      Reason:       OOMKilled
      Exit Code:    137
      Started:      Tue, 26 Sep 2023 17:57:16 +0530
      Finished:     Tue, 26 Sep 2023 18:03:32 +0530
    Ready:          True
    Restart Count:  1
    Environment:    <none>
    Mounts:
      /var/run/secrets/kubernetes.io/serviceaccount from kube-api-access-fhxqs (ro)
Conditions:
  Type              Status
  Initialized       True
  Ready             True
  ContainersReady   True
  PodScheduled      True
Volumes:
  kube-api-access-fhxqs:
    Type:                    Projected (a volume that contains injected data from multiple sources)
    TokenExpirationSeconds:  3607
    ConfigMapName:           kube-root-ca.crt
    ConfigMapOptional:       <nil>
    DownwardAPI:             true
QoS Class:                   BestEffort
Node-Selectors:              <none>
Tolerations:                 node.kubernetes.io/not-ready:NoExecute op=Exists for 300s
                             node.kubernetes.io/unreachable:NoExecute op=Exists for 300s
Events:
  Type     Reason        Age                    From               Message
  ----     ------        ----                   ----               -------
  Normal   Scheduled     9m38s                  default-scheduler  Successfully assigned default/clip-classifer-deployment-5cc8894f98-lc4v2 to minikube
  Normal   Pulled        3m20s (x2 over 9m37s)  kubelet            Container image "clip-k8s:latest" already present on machine
  Normal   Created       3m20s (x2 over 9m37s)  kubelet            Created container clip-classifier
  Normal   Started       3m20s (x2 over 9m37s)  kubelet            Started container clip-classifier
  Warning  NodeNotReady  35s (x2 over 6m19s)    node-controller    Node is not ready
```

3. kubectl describe <your_ingress>

```
Name:             clip-classifier-ingress
Labels:           <none>
Namespace:        default
Address:
Ingress Class:    <none>
Default backend:  <default>
Rules:
  Host                       Path  Backends
  ----                       ----  --------
  clip-classifier.localhost
                             /   clip-classifier-service:80 (10.244.0.36:7860)
Annotations:                 <none>
Events:                      <none>
```

4. kubectl top pod

```
NAME                                         CPU(cores)   MEMORY(bytes)
classifer-deployment-86bb8c55f-jsqdb         427m         529Mi
classifer-deployment-86bb8c55f-k2wfc         421m         524Mi
clip-classifer-deployment-5cc8894f98-lc4v2   337m         540Mi
clip-classifer-deployment-5cc8894f98-n8cks   336m         542Mi
```

5. kubectl top node

```
NAME       CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%
minikube   1005m        25%    3466Mi          44%  
```

6. kubectl get all -A -o yaml

[Get all Details](https://github.com/shariqfarhan/EMLOV3/blob/Main/Assignment_16/Assignment/backup.yaml)
