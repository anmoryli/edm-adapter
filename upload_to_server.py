"""Fast upload to GPU server via paramiko with optimized buffer sizes"""
import paramiko
import os
import sys
import time

host = '36.111.81.182'
port = 50000
username = 'root'
password = '.taawwm8lf5g'

local_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'edm-adapter.zip'))
remote_path = '/root/edm-adapter.zip'

if not os.path.exists(local_path):
    print(f'ERROR: File not found: {local_path}', flush=True)
    sys.exit(1)
file_size = os.path.getsize(local_path)
print(f'Local file: {local_path} ({file_size / (1024**3):.2f} GB)', flush=True)

print(f'Connecting to {host}:{port}...', flush=True)
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Use larger window size for faster transfers
ssh.connect(
    host, port=port, username=username, password=password,
    banner_timeout=60, auth_timeout=60,
)
print('SSH connected!', flush=True)

transport = ssh.get_transport()
# Increase window size for better throughput
transport.window_size = 2 * 1024 * 1024  # 2MB window

sftp = paramiko.SFTPClient.from_transport(transport)

# Set larger packet size
sftp.MAX_REQUEST_SIZE = 1024 * 1024  # 1MB packets

start_time = time.time()
last_print = [0]
last_bytes = [0]
speed_samples = []

def progress_callback(bytes_transferred, total_bytes):
    now = time.time()
    if now - last_print[0] >= 3:  # Print every 3 seconds
        elapsed_since_last = now - last_print[0]
        bytes_since_last = bytes_transferred - last_bytes[0]
        instant_speed = bytes_since_last / elapsed_since_last / (1024 * 1024)

        last_print[0] = now
        last_bytes[0] = bytes_transferred

        speed_samples.append(instant_speed)
        # Keep last 5 samples for average
        if len(speed_samples) > 5:
            speed_samples.pop(0)
        avg_speed = sum(speed_samples) / len(speed_samples)

        percent = (bytes_transferred / total_bytes) * 100
        mb_done = bytes_transferred / (1024**2)
        remaining_bytes = total_bytes - bytes_transferred
        eta_seconds = remaining_bytes / (avg_speed * 1024 * 1024) if avg_speed > 0 else 0

        print(f'Progress: {percent:.1f}% ({mb_done:.0f}/{total_bytes/(1024**2):.0f} MB) | '
              f'Speed: {avg_speed:.1f} MB/s | ETA: {eta_seconds/60:.0f} min', flush=True)

print(f'Uploading to {remote_path}...', flush=True)

# Use a buffered file reader for better performance
with open(local_path, 'rb') as f:
    sftp.putfo(f, remote_path, file_size=file_size, callback=progress_callback)

elapsed = time.time() - start_time
avg_speed = file_size / elapsed / (1024 * 1024) if elapsed > 0 else 0
print(f'\nUpload complete! Time: {elapsed/60:.1f} min, Average speed: {avg_speed:.1f} MB/s', flush=True)

remote_size = sftp.stat(remote_path).st_size
print(f'Remote file size: {remote_size / (1024**3):.2f} GB', flush=True)
print(f'Verification: {"PASS" if remote_size == file_size else "FAIL"}', flush=True)

sftp.close()
ssh.close()
print('Done!', flush=True)
