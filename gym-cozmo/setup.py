from setuptools import setup

setup(name='gym_cozmo',
      version='0.0.1',
      install_requires=[
          'gym',
          'cozmo[camera]'
      ]
)