import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('1.14.150.130', port=22, username='root', password='Lmjnb689####', timeout=10)

# Kill the specific process
cmds = [
    'kill -9 4050414 2>/dev/null',
    'sleep 1',
    'ss -tlnp | grep -E "17860|27860" || echo "ports free"',
    'systemctl restart ssh',
    'sleep 1',
    'ss -tlnp | grep -E "17860|27860" || echo "ports free after restart"',
]
for cmd in cmds:
    stdin, stdout, stderr = c.exec_command(cmd, timeout=15)
    stdout.channel.recv_exit_status()
    out = stdout.read().decode().strip()
    if out:
        print(out)
c.close()
