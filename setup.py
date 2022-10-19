import os
from setuptools import setup, find_packages


def run_setup():
    # get version
    version_namespace = {}
    with open(os.path.join('nicr_detectron2', 'version.py')) as version_file:
        exec(version_file.read(), version_namespace)
    version = version_namespace['_get_version'](with_suffix=False)

    # setup
    setup(name='nicr_detectron2',
          version='{}.{}.{}'.format(*version),
          description='Package containing architectures for detectron2.',
          author='Benedict Stephan',
          author_email='benedict.stephan@tu-ilmenau.de',
          install_requires=[
              'numpy',
              'torch',
              'torchvision',
              'detectron2 @ git+https://github.com/facebookresearch/detectron2.git#egg=detectron2',
              'opencv-python',
              'scikit-image'
          ],
          packages=find_packages())


if __name__ == '__main__':
    run_setup()
