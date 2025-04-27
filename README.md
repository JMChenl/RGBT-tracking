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

# Code Reference
The code is based on the pysot library：https://github.com/StrangerZhang/pysot-toolkit
Reference：https://github.com/ohhhyeahhh/SiamCAR和
