"""SSH reverse tunnel - binds to 0.0.0.0 on remote."""
import subprocess

REMOTE_HOST = '1.14.150.130'
LOCAL_PORT = 7860
REMOTE_PORT = 17860

ssh_cmd = [
    'ssh', '-o', 'StrictHostKeyChecking=no',
    '-o', 'ServerAliveInterval=30',
    '-o', 'ServerAliveCountMax=3',
    '-N',
    '-R', f'0.0.0.0:{REMOTE_PORT}:127.0.0.1:{LOCAL_PORT}',
    f'root@{REMOTE_HOST}',
]
print(f'Tunnel: {REMOTE_HOST}:{REMOTE_PORT} -> 127.0.0.1:{LOCAL_PORT}')
print(f'Access: http://{REMOTE_HOST}:{REMOTE_PORT}')
print('Press Ctrl+C to stop.')
subprocess.run(ssh_cmd)
