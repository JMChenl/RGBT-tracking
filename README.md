# Virtual environment installation
virtualenv -p python3.6 venv
source ./venv/bin/activate

# Install library dependencies
pip install -r requiremenets.txt

# model train
cd tools
python train.py

# model test
cd tools
python test.py

# Datasets
GTOT：https://pan.baidu.com/s/1QNidEo-HepRaS6OIZr7-Cw
RGBT234：https://drive.google.com/file/d/1ouNEptXOgRop4U7zYMK9zAp57SZ2XCNL/view
LasHeR:https://pan.baidu.com/share (Password:mmic)

# Code Reference
The code is based on the pysot library：https://github.com/StrangerZhang/pysot-toolkit
Reference：https://github.com/ohhhyeahhh/SiamCAR and https://github.com/easycodesniper-afk/SiamCSR
