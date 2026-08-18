# 01-dockers

昇腾训练容器启动脚本。

## start_container.sh

启动 verl 训练容器（host 网络、特权模式、挂载 NPU driver / `npu-smi` / `/home`）。

```bash
bash 01-tools/01-dockers/start_container.sh
```

默认容器名：`verl_0526`  
默认镜像：`quay.io/ascend/verl:verl-8.5.0-910b-ubuntu22.04-py3.11-latest`

如需换镜像，修改脚本末尾的 image 行；脚本内注释了若干历史镜像 tag 备查。
