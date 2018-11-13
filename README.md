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

96eyesnose
python ./util/align-dlib.py ../assets/db/original/biometix-morphed/imgs/ align outerEyesAndNose ../assets/db/96eyesnose/biometix-morphed/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/96eyesnose/biometix-morphed-csv-rep -data ../assets/db/96eyesnose/biometix-morphed
python ./util/align-dlib.py ../assets/db/original/biometix-genuine/imgs/ align outerEyesAndNose ../assets/db/96eyesnose/biometix-genuine/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/96eyesnose/biometix-genuine-csv-rep -data ../assets/db/96eyesnose/biometix-genuine
python ./util/align-dlib.py ../assets/db/original/feret-genuine/imgs/ align outerEyesAndNose ../assets/db/96eyesnose/feret-genuine/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/96eyesnose/feret-genuine-csv-rep -data ../assets/db/96eyesnose/feret-genuine

256eyesnose
python ./util/align-dlib.py ../assets/db/original/biometix-morphed/imgs/ align outerEyesAndNose ../assets/db/256eyesnose/biometix-morphed/imgs --size 256
python ./util/align-dlib.py ../assets/db/original/biometix-genuine/imgs/ align outerEyesAndNose ../assets/db/256eyesnose/biometix-genuine/imgs --size 256
python ./util/align-dlib.py ../assets/db/original/feret-genuine/imgs/ align outerEyesAndNose ../assets/db/256eyesnose/feret-genuine/imgs --size 256

96eyeslip
python ./util/align-dlib.py ../assets/db/original/biometix-morphed/imgs/ align innerEyesAndBottomLip ../assets/db/96eyeslip/biometix-morphed/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/96eyeslip/biometix-morphed-csv-rep -data ../assets/db/96eyeslip/biometix-morphed
python ./util/align-dlib.py ../assets/db/original/biometix-genuine/imgs/ align innerEyesAndBottomLip ../assets/db/96eyeslip/biometix-genuine/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/96eyeslip/biometix-genuine-csv-rep -data ../assets/db/96eyeslip/biometix-genuine
python ./util/align-dlib.py ../assets/db/original/feret-genuine/imgs/ align innerEyesAndBottomLip ../assets/db/96eyeslip/feret-genuine/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/96eyeslip/feret-genuine-csv-rep -data ../assets/db/96eyeslip/feret-genuine

256eyeslip
python ./util/align-dlib.py ../assets/db/original/biometix-morphed/imgs/ align innerEyesAndBottomLip ../assets/db/256eyeslip/biometix-morphed/imgs --size 256 
python ./util/align-dlib.py ../assets/db/original/biometix-genuine/imgs/ align innerEyesAndBottomLip ../assets/db/256eyeslip/biometix-genuine/imgs --size 256 
python ./util/align-dlib.py ../assets/db/original/feret-genuine/imgs/ align innerEyesAndBottomLip ../assets/db/256eyeslip/feret-genuine/imgs --size 256 


------------------------------------------------------------

96eyesnose
python ./../openface/util/align-dlib.py ../assets/db/digital/original/morphed/imgs/ align outerEyesAndNose ../assets/db/digital/96eyesnose/morphed/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/digital/96eyesnose/morphed-csv-rep -data ../assets/db/digital/96eyesnose/morphed
python ./../openface/util/align-dlib.py ../assets/db/digital/original/genuine4morphed/imgs/ align outerEyesAndNose ../assets/db/digital/96eyesnose/genuine4morphed/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/digital/96eyesnose/genuine4morphed-csv-rep -data ../assets/db/digital/96eyesnose/genuine4morphed
python ./../openface/util/align-dlib.py ../assets/db/digital/original/genuine/imgs/ align outerEyesAndNose ../assets/db/digital/96eyesnose/genuine/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/digital/96eyesnose/genuine-csv-rep -data ../assets/db/digital/96eyesnose/genuine

256eyesnose
python ./../openface/util/align-dlib.py ../assets/db/digital/original/morphed/imgs/ align outerEyesAndNose ../assets/db/digital/256eyesnose/morphed/imgs --size 256
python ./../openface/util/align-dlib.py ../assets/db/digital/original/genuine4morphed/imgs/ align outerEyesAndNose ../assets/db/digital/256eyesnose/genuine4morphed/imgs --size 256
python ./../openface/util/align-dlib.py ../assets/db/digital/original/genuine/imgs/ align outerEyesAndNose ../assets/db/digital/256eyesnose/genuine/imgs --size 256

96eyeslip
python ./../openface/util/align-dlib.py ../assets/db/digital/original/morphed/imgs/ align innerEyesAndBottomLip ../assets/db/digital/96eyeslip/morphed/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/digital/96eyeslip/morphed-csv-rep -data ../assets/db/digital/96eyeslip/morphed
python ./../openface/util/align-dlib.py ../assets/db/digital/original/genuine4morphed/imgs/ align innerEyesAndBottomLip ../assets/db/digital/96eyeslip/genuine4morphed/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/digital/96eyeslip/genuine4morphed-csv-rep -data ../assets/db/digital/96eyeslip/genuine4morphed
python ./../openface/util/align-dlib.py ../assets/db/digital/original/genuine/imgs/ align innerEyesAndBottomLip ../assets/db/digital/96eyeslip/genuine/imgs --size 96 && luajit ./batch-represent/main.lua -outDir ../assets/db/digital/96eyeslip/genuine-csv-rep -data ../assets/db/digital/96eyeslip/genuine

256eyeslip
python ./../openface/util/align-dlib.py ../assets/db/digital/original/morphed/imgs/ align innerEyesAndBottomLip ../assets/db/digital/256eyeslip/morphed/imgs --size 256 
python ./../openface/util/align-dlib.py ../assets/db/digital/original/genuine4morphed/imgs/ align innerEyesAndBottomLip ../assets/db/digital/256eyeslip/genuine4morphed/imgs --size 256 
python ./../openface/util/align-dlib.py ../assets/db/digital/original/genuine/imgs/ align innerEyesAndBottomLip ../assets/db/digital/256eyeslip/genuine/imgs --size 256 



file lua per embeddings in folder:
GUIDA https://gist.github.com/ageitgey/ddbae3b209b6344a458fa41a3cf75719
.LUA  https://github.com/cmusatyalab/openface/blob/master/batch-represent/batch-represent.lua