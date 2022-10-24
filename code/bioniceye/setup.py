from setuptools import setup

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
  name='bioniceye',
  version='1.0.0',
  description='A Novel Environment',
  author='Na Min An',
  author_email='namin0202@gmail.com',
  keywords='environment',
  url = "https://github.com/namin-an",
  packages=['bioniceye'],
  install_requires=['glob', 'numpy', 'matplotlib', 'pandas', 'opencv-python', 'gym', 'torch', 'torchvision', 'torchaudio']
)
