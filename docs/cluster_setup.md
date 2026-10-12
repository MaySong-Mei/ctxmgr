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

## On the cluster

Repo lives in the course project dir: `/nfs/roberts/project/<course>/<course>_<netid>/ctxmgr`.
All computation goes through Slurm (`sbatch slurm/...`) — never on login nodes.

```bash
mkdir -p logs
sbatch slurm/setup_env.sbatch Qwen/Qwen3.5-0.8B   # CPU node: uv, vLLM, weights
sbatch slurm/serve_smoke.sbatch Qwen/Qwen3.5-0.8B # 1 GPU: serve + smoke test
squeue -u $USER
```
