#!/usr/bin/env bash
# One-time Bouchet SSH setup inside WSL (Ubuntu). See docs/cluster_setup.md.
# Run from Ubuntu:  bash /mnt/c/.../ctxmgr/scripts/setup_wsl_ssh.sh
# Asks for the cluster username (<course>_<netid>) and a key passphrase; writes nothing to the repo.
set -euo pipefail

ALIAS=bouchet-cpsc4880
KEY="$HOME/.ssh/id_ed25519_ycrc"

sudo apt-get update -qq
sudo apt-get install -y -qq openssh-client expect netcat-openbsd

mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"

if [ ! -f "$KEY" ]; then
  echo "Generating a new key for this device. Set a passphrase when asked (Bouchet requires one)."
  ssh-keygen -t ed25519 -f "$KEY" -C "$(whoami)@$(hostname)-wsl"
fi

if ! grep -q "^Host $ALIAS\$" "$HOME/.ssh/config" 2>/dev/null; then
  read -rp "Cluster username (<course>_<netid>): " CLUSTER_USER
  cat >> "$HOME/.ssh/config" <<EOF

Host $ALIAS
  HostName bouchet.ycrc.yale.edu
  User $CLUSTER_USER
  IdentityFile $KEY
  IdentitiesOnly yes
  AddKeysToAgent yes
  ServerAliveInterval 60
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 4h
EOF
  chmod 600 "$HOME/.ssh/config"
fi

echo
echo "Public key — paste it at https://sshkeys.ycrc.yale.edu/ (Yale VPN on), then wait ~10 min:"
cat "$KEY.pub"
echo
if nc -z -w 8 bouchet.ycrc.yale.edu 22; then
  echo "Bouchet port 22 reachable."
else
  echo "Bouchet not reachable: connect Yale VPN on Windows and retry."
fi
echo "First login (Duo push):  ssh $ALIAS"
