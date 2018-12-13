# Master Thesis

## Technologies

- [Dlib](#Dlib)


## Installation

### Dlib
Source: [Instructions](https://www.learnopencv.com/install-dlib-on-macos/)

#### OS libraries

`brew cask install xquartz`

`brew install gtk+3 boost`

`brew install boost-python --with-python3`

#### Python 3

`brew install python python3`

`brew link python`

`brew link python3`

#### Python libraries

**NB** Inside `venv`

`pip install numpy scipy matplotlib scikit-image scikit-learn ipython`

#### Install Dlib

`brew install dlib`

**NB** Inside `venv`

`pip install dlib`

#### Install X11

Go [here](https://www.xquartz.org/), follow installation instructions and log out and log back in.



git clone https://github.com/cmusatyalab/openface.git --recursive 

pip install opencv-python

brew install wget
brew install findutils
brew install coreutils

git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
source ~/.zshrc

for NAME in dpnn nn optim optnet csvigo cutorch cunn fblualib torchx tds; do luarocks install $NAME; done   

cd openface

python setup.py install

./models/get-models.sh

Lua file for embeddings in folder:
GUIDA https://gist.github.com/ageitgey/ddbae3b209b6344a458fa41a3cf75719
.LUA  https://github.com/cmusatyalab/openface/blob/master/batch-represent/batch-represent.lua
