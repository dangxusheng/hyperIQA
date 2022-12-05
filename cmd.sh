
################## python environment ##################
system:
    ubuntu 18.04
    nvidia==470.129.06
    cuda==11.4
conda env:
    cudatoolkit=11.3.1
    cudnn==8.2.1
    python==3.7
    opencv-python==4.2.0.32
    mmcv==1.5.0
    mmdetection==2.25.0
    mmcls==0.23.1
    mmsegmentation==0.25.0
    mmrotate==0.3.2
    pytorch==1.10.0
    torchvision==0.11.2
######################################################


################## 10.0.13.200:22   sunnypc/123  ##################
ssh -p 22 sunnypc@10.0.13.200
ssh  -p 22 -o  ServerAliveInterval=60 sunnypc@10.0.13.200
123


cd /home/sunnypc/dangxs/projects/python_projects/IQA-code/hyperIQA-master && \
conda activate py37_torch180_cu113


# windows写好sh送到服务端执行的时候有时候会提示 找不到文件
# 原因是sh文件的编码有问题，多出了字符导致不识别
# 第一步：先查看sh文件是否有^M
cat -A compare-tcp-algorithms.sh
# 第二部：去除字符
sed -i 's/\r$//' compare-tcp-algorithms.sh



# 后台运行
nohup sh ./tools/train.sh > run_nohup.log 2>&1 &


# 找到nohup启动运行的pid   12262
ps -aux | grep "train.sh"
kill -9  进程号PID


find / -name "libprotobuf.so*" | xargs grep -n "libprotobuf.so*"
find / -name "libre2.so*" | xargs grep -n "libre2.so*"



# downlaod mmcv url:  https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html


# jupyter notebook 启动
jupyter notebook --ip=0.0.0.0 --allow-root
nohup  jupyter notebook --ip=0.0.0.0 --allow-root  >>out.log 2>&1 &

http://10.0.13.200:8870     123

# 查看端口占用
sudo lsof -i:8870
kill -9 PID
