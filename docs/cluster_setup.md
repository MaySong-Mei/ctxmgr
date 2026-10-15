# Connecting to Bouchet (course account)

Per the Bouchet Rules of Behavior: one key per device, protected by a passphrase; never copy a
private key between machines; devices must auto-lock after at most 45 minutes of inactivity.

## On each new machine (macOS)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_ycrc -C "<netid>@<device>"   # set a passphrase
pbcopy < ~/.ssh/id_ed25519_ycrc.pub
```

Paste the public key at https://sshkeys.ycrc.yale.edu/ (on campus or Yale VPN). Existing keys stay
valid; propagation takes up to ~10 minutes.

Add to `~/.ssh/config` (replace `<course>_<netid>`):

```
Host bouchet
  HostName bouchet.ycrc.yale.edu
  User <course>_<netid>
  IdentityFile ~/.ssh/id_ed25519_ycrc
  IdentitiesOnly yes
  AddKeysToAgent yes
  UseKeychain yes
  ServerAliveInterval 60
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 4h
```

Then `ssh-add --apple-use-keychain ~/.ssh/id_ed25519_ycrc` and `ssh bouchet` once (Duo push).
`ControlPersist` keeps that authenticated connection for 4 h of idle time, so later
non-interactive `ssh bouchet <cmd>` calls skip Duo.

## On a Windows machine (via WSL)

Windows' built-in OpenSSH does not support `ControlMaster`/`ControlPersist`, and the agent workflow
also needs `expect` and `nc`. Use WSL (Ubuntu) and treat it as its own device with its own key:

```powershell
wsl --install -d Ubuntu        # once, from an admin PowerShell; reboot if asked
```

Inside Ubuntu:

```bash
sudo apt update && sudo apt install -y openssh-client expect netcat-openbsd
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_ycrc -C "<netid>@<device>-wsl"   # set a passphrase
cat ~/.ssh/id_ed25519_ycrc.pub   # paste at https://sshkeys.ycrc.yale.edu/
```

Use the same `~/.ssh/config` block as above, minus `UseKeychain yes` (macOS only). Load the key
with `eval "$(ssh-agent -s)" && ssh-add ~/.ssh/id_ed25519_ycrc`. The reachability check differs:
`nc -z -w 8 bouchet.ycrc.yale.edu 22` (`-G` is macOS-only). Yale VPN must be connected on Windows;
WSL shares it.

## On the cluster

Repo lives in the course project dir: `/nfs/roberts/project/<course>/<course>_<netid>/ctxmgr`.
All computation goes through Slurm (`sbatch slurm/...`) — never on login nodes.

```bash
mkdir -p logs
sbatch slurm/setup_env.sbatch Qwen/Qwen3.5-0.8B   # CPU node: uv, vLLM, weights
sbatch slurm/serve_smoke.sbatch Qwen/Qwen3.5-0.8B # 1 GPU: serve + smoke test (several models OK)
squeue -u $USER
```
