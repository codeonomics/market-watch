mkdir Analysis
mkdir Analysis/{code,scripts,data}
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash ~/Anaconda3-2024.10-1-Linux-x86_64.sh
which python
source .bashrc 
which python
which pip
pip install dash
pip install pyxirr
cd Analysis/scripts/
nohup python app.py > app.log 2>&1 &
ps aux | grep python
sudo dnf install git-all