# Application Dev
## Prerequisties
```
sudo apt update
sudo apt upgrade -y
sudo apt autoremove -y
sudo apt install git -y
sudo apt install docker.io
```
## Install and Start Server
```
git clone -b application-dev --single-branch https://github.com/smithandrewk/aurora.git
cd aurora
make docker
```
After running these commands, you should find the web app at http://localhost.
