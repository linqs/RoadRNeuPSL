# Data

## Frame-Extraction

Before extracting the frames, you will need to make sure that you have `ffmpeg` installed on your machine or your 
python should include its binaries. If you are using Ubuntu, the following command should be sufficient: 
`sudo apt install ffmpeg`.

## AWS Setup

This should be ignored if not setting up from scratch on AWS.

### Setting up ffmpeg
`cd /usr/local/bin`

`mkdir ffmpeg && cd ffmpeg`

`wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz`

`tar -xf ffmpeg-release-amd64-static.tar.xz`

`cp -a /usr/local/bin/ffmpeg/ffmpeg-6.0-amd64-static/. .`

`ln -s /usr/local/bin/ffmpeg/ffmpeg /usr/bin/ffmpeg`

`exit`

### Install pip3 packages
`pip3 install numpy torch transformers pillow scipy torchvision matplotlib torchmetrics pycocotools timm git-python`

### Generate SSH keys
`cd ~/.ssh`

`ssh-keygen -t ed25519 -C "<username>@<domain>"`

`cat id_ed25519.pub`

### Clone git
`cd ~ && mkdir work && cd work`

`git clone git@github.com:linqs/RoadRNeuPSL.git`

`cd RoadRNeuPSL/data`

`./fetchData.sh`

`python3 videos_to_jpgs.py`

`cd ../experiments/`

`python3 task1_pretrain.py`

### Install mvn
`cd ~/work/`

`wget https://dlcdn.apache.org/maven/maven-3/3.9.5/binaries/apache-maven-3.9.5-bin.tar.gz`

`tar xzvf apache-maven-3.9.5-bin.tar.gz`

### Update ~/.bashrc
`export JAVA_HOME="/usr/lib/jvm/java-11-openjdk"`

`export PATH="/usr/bin/java":$PATH`

`export PATH="/home/ec2-user/work/apache-maven-3.9.5/bin/":$PATH`

`export LD_LIBRARY_PATH="/home/ec2-user/work/apache-maven-3.9.5/lib/":$LD_LIBRARY_PATH`

### Update .bashrc
`source ~/.bashrc`