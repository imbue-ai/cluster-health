#!/bin/bash

export DEBIAN_FRONTEND=noninteractive
export DEBCONF_NONINTERACTIVE_SEEN=true


echo 'deb https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH) /' | sudo tee -a /etc/apt/sources.list.d/nvidia_github_io_libnvidia_container_stable_deb_ARCH.list
echo 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /' | sudo tee -a  /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && yes | sudo dpkg -i cuda-keyring_1.0-1_all.deb
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
#sudo apt-get update -y


sudo apt-get install -y linux-headers-$(uname -r)
#sudo apt-get install -y linux-nvidia=5.15.0.1042.42 linux-tools-nvidia=5.15.0.1042.42 --no-install-recommends
sudo apt-get install -y linux-nvidia=5.15.0.1047.47 linux-tools-nvidia=5.15.0.1047.47 --no-install-recommends --allow-downgrades

#sudo apt-get install -s cuda-drivers-fabricmanager-535=535.183.01-1   nvidia-fabricmanager-535=535.183.01-1  cuda-drivers-535=535.183.01-1 --no-install-recommends
#sudo apt-get install -y cuda-drivers-fabricmanager-535=535.183.01-1   nvidia-fabricmanager-535=535.183.01-1  cuda-drivers-535=535.183.01-1 --no-install-recommends

sudo apt-get install -y --no-install-recommends --allow-downgrades \
	cuda-drivers-535=535.183.01-1 \
	cuda-drivers-fabricmanager-535=535.183.01-1 \
	libnvidia-cfg1-535:amd64=535.183.01-0ubuntu1 \
	libnvidia-common-535=535.183.01-0ubuntu1 \
	libnvidia-compute-535:amd64=535.183.01-0ubuntu1 \
	libnvidia-decode-535:amd64=535.183.01-0ubuntu1 \
	libnvidia-encode-535:amd64=535.183.01-0ubuntu1 \
	libnvidia-extra-535:amd64=535.183.01-0ubuntu1 \
	libnvidia-fbc1-535:amd64=535.183.01-0ubuntu1 \
	libnvidia-gl-535:amd64=535.183.01-0ubuntu1 \
	nvidia-compute-utils-535=535.183.01-0ubuntu1 \
	nvidia-dkms-535=535.183.01-0ubuntu1 \
	nvidia-driver-535=535.183.01-0ubuntu1 \
	nvidia-fabricmanager-535=535.183.01-1 \
	nvidia-kernel-common-535=535.183.01-0ubuntu1 \
	nvidia-kernel-source-535=535.183.01-0ubuntu1 \
	nvidia-utils-535=535.183.01-0ubuntu1 \
	xserver-xorg-video-nvidia-535=535.183.01-0ubuntu1

sudo sed -i 's/--no-persistence-mode/--persistence-mode/' /usr/lib/systemd/system/nvidia-persistenced.service
sudo systemctl enable nvidia-fabricmanager.service
sudo apt-get install -y nvidia-container-toolkit=1.14.3-1 nvidia-container-toolkit-base=1.14.3-1

sudo modprobe nvidia
sudo service docker restart
sudo modprobe nvidia-peermem || true
