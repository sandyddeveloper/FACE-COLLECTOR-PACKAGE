from setuptools import setup, find_packages

setup(
    name='face_collector',
    version='0.1.0',
    description='A package to collect and align faces from a video stream',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'numpy',
        'Pillow',
        'torch',
        'requests',
        'facenet-pytorch'
    ],
    entry_points={
        'console_scripts': [
            'face-collector=face_collector.main:main',
        ],
    },
)
