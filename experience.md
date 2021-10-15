1. 安装nvidia驱动
2. 安装cuda
    - 各个版本的cuda：https://developer.nvidia.com/cuda-toolkit-archive
    - 直接安装需要的cuda版本，不需要考虑过版本会冲突的问题，安装好后会在`usr/local/bin/`下。
    - 如果已经安装了驱动，安装cuda要取消安装驱动。
    - home文件夹下的`.bashrc`文件可以切换不同的cuda。
3. 安装cudnn：https://developer.nvidia.com/rdp/cudnn-archive
4. 安装pytorch：https://pytorch.org/get-started/previous-versions/
5. pip install时报错InvalidSchema: Missing dependencies for SOCKS support.
   - export all_proxy=""
6. 在shell脚本使用conda虚拟环境
   - 在base环境下执行：source activate env_name
   - 然后使用source执行shell脚本
- 