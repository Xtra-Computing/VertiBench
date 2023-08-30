ifconfig eno1 > begin.txt;
time GLOO_DEBUG=INFO TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=INFO GLOO_SOCKET_IFNAME=eno1 torchrun --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr=192.168.47.111 --master_port=12347 src/algorithm/DistSplitNN.py -d covtype -c 7 -m acc -p 4 -sp imp -w 0.1 -s 0 -g 0 > log.txt 2>&1
ifconfig eno1 > end.txt;
