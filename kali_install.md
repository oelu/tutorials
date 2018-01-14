# Kali Installation

## Basic install

1. Download and install Kali. 
2. Update system: `apt-get update && apt-get dist-upgrade -y`

## Install packages

1. Install packages from repo: https://github.com/oelu/dotfiles/tree/master/bootstrap

## User setup

1. `useradd -m username`
2. `passwd username`
3. `usermod -a -G sudo username`
4. `chsh /usr/bin/zsh username`
5. Install `oh-my-zsh` from https://github.com/robbyrussell/oh-my-zsh
6. Install `spf13-vim` from https://github.com/spf13/spf13-vim
7. Copy home directory from previous install

## Install Sublime Text

Follow instructions at: https://www.sublimetext.com/docs/3/linux_repositories.html

## SSH daemon

1. `update-rc.d -f ssh remove`
2. `update-rc.d -f ssh defaults`
3. `cd /etc/ssh`
4. `mkdir insecure_original_default_kali_keys`
5. `mv ssh_host_* insecure_original_default_kali_keys`
6. `dpkg-reconfigure openssh-server`
7. `service ssh restart`
8. `update-rc.d -f ssh enable 2 3 4 5`

### SSHD configuration

* Disable password authentication in `/etc/ssh/sshd_config`.

```
PasswordAuthentication no
```

* Disable remote login for `root` in `/etc/ssh/sshd_config`.

```
PermitRootLogin no
```
