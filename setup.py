from setuptools import setup

setup(
    name='image2pipe-nicholas-tancredi',
    version='0.1.3',
    author='Nicholas Tancredi',
    keywords="ffmpeg yuv image2pipe",
    packages=['image2pipe', ],
    scripts=[],
    license='LICENSE.txt',
    description='Simple ffmpeg wrapper for image2pipe which yields rawvideo frames from input video URL',
    long_description_content_type="text/markdown",
    install_requires=['tblib', 'numpy', 'websocket'],
    python_requires='>=3.2',
)
