import subprocess

if __name__ == '__main__':
    width = 358
    height = 656
    _proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, bufsize=10 ** 8)
    img_size = width * height * 3
    bb = _proc.stdout.read(img_size)