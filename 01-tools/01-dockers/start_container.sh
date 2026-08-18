container_name=verl_0526

docker run -itd \
    --shm-size 10g \
    --network host \
    --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
    -v /home/:/home \
    --name $container_name \
    quay.io/ascend/verl:verl-8.5.0-910b-ubuntu22.04-py3.11-latest
    # jiutian236b:v3
    # quay.io/ascend/verl:verl-8.5.0-910b-ubuntu22.04-py3.11-latest
    # quay.io/ascend/verl:verl-8.3.rc1-910b-ubuntu22.04-py3.11-v0.7.0
    # quay.io/ascend/verl:verl-8.5.0-910b-ubuntu22.04-py3.11-latest
    # quay.io/ascend/vllm-ascend:v0.13.0rc1
