1. 安装nvidia驱动
2. 安装cuda
    - 各个版本的cuda：https://developer.nvidia.com/cuda-toolkit-archive
    - cuda与gcc的版本对应：https://docs.nvidia.com/cuda/archive/10.2/cuda-installation-guide-linux/index.html ，把网址中的cuda版本号换成想要查询的
    - 直接使用`sudo sh`安装需要的cuda版本的runfile，不需要考虑过版本会冲突的问题，安装好后会在`usr/local/bin/`下。
    - 如果已经安装了驱动，安装cuda要取消安装驱动。
    - home文件夹下的`.bashrc`文件可以切换不同的cuda。
    - 通过`nvcc -V`查看cuda版本
3. 安装cudnn：https://developer.nvidia.com/rdp/cudnn-archive
   - 下载cudnn后解压，进入目录后执行：
     - sudo cp cuda/include/cudnn.h /usr/local/cuda-10.1/include
     - sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.1/lib64
     - sudo chmod a+r /usr/local/cuda-10.1/include/cudnn.h 
     - sudo chmod a+r /usr/local/cuda-10.1/lib64/libcudnn*
   - 使用下述命令查看cudnn版本
     - cat /usr/local/cuda-10.1/include/cudnn.h | grep CUDNN_MAJOR -A 2
4. 安装pytorch：https://pytorch.org/get-started/previous-versions/
5. pip install时报错InvalidSchema: Missing dependencies for SOCKS support.
   - export all_proxy=""
6. 在shell脚本使用conda虚拟环境
   - 在base环境下执行：source activate env_name
   - 然后使用source执行shell脚本
- 