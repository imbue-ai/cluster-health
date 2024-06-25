sudo lsof /dev/nvidia* | awk '{ print $2 }' | grep -v PID  | uniq | xargs sudo kill -9

timeout 5 sudo systemctl stop nvidia-persistenced &
sleep 5
sudo lsof /dev/nvidia* | awk '{ print $2 }' | grep -v PID  | uniq | xargs sudo kill -9
sudo rmmod nvidia_drm nvidia_modeset nvidia nvidia_uvm
sudo rmmod nvidia_drm nvidia_modeset nvidia nvidia_uvm
sudo rmmod nvidia_drm nvidia_modeset nvidia nvidia_uvm
sudo rmmod nvidia_drm nvidia_modeset nvidia nvidia_uvm
sudo rmmod nvidia_drm nvidia_modeset nvidia nvidia_uvm
sudo modprobe -r nvidia
sudo rm -r /usr/share/nvidia/nvswitch
sudo apt remove -y --purge '^nvidia-.*'
sudo apt purge -y --auto-remove '^nvidia-.*'