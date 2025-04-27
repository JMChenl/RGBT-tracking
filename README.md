# 虚拟环境安装
virtualenv -p python3.6 venv
source ./venv/bin/activate
# 安装库依赖
pip install -r requiremenets.txt

# 模型训练
cd到tools目录下，运行命令python train.py

# 模型测试
cd到tools目录下，运行命令python test.py
