import subprocess

while True:
    print('Starting...')
    process = subprocess.Popen(['python', 'main.py'])
    # 替换'your_script.py'为你的实际脚本文件名
    exit_code = process.wait()
    print('Finished with exit code:', exit_code)
    if exit_code == 0 or exit_code == -6:
        print('Restarting...')
    else:
        print('Error occurred, not restarting...')
        break