import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('1.14.150.130', port=22, username='root', password='Lmjnb689####', timeout=10)
cmds = [
    'ss -tlnp | grep 27860',
    'ss -tlnp | grep 17860',
    'curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7860',
    'iptables -L INPUT -n | head -20',
    'ufw status 2>/dev/null || echo no-ufw',
]
for cmd in cmds:
    stdin, stdout, stderr = c.exec_command(cmd, timeout=10)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f'CMD: {cmd}')
    print(f'  OUT: {out}')
    if err:
        print(f'  ERR: {err}')
c.close()
