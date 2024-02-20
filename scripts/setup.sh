pip3 --no-cache-dir install \
     torch==1.11.0+cu113 \
     torchvision==0.12.0+cu113 \
     --extra-index-url https://download.pytorch.org/whl/cu113

pip3 install natten==0.14.4

pip install -r ./requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install av2 -i https://pypi.tuna.tsinghua.edu.cn/simple