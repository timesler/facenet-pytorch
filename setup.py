import setuptools

with open('facenet_pytorch/README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='facenet-pytorch',
    version='0.1.0',
    author='Tim Esler',
    author_email='tim.esler@gmail.com',
    description='Pretrained Pytorch face detection and recognition models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/timesler/facenet-pytorch',
    packages=[
        'facenet_pytorch',
        'facenet_pytorch.models',
        'facenet_pytorch.models.utils',
        'facenet_pytorch.data',
   ],
    package_data={'': ['*net.pt']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'requests',
        'opencv-python',
    ],
)
