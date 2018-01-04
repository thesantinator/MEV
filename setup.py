
from setuptools import setup

setup(
   name='mevpy',
   version='1.01',
   description='mevpy package',
   author='Enrico Zorzetto',
   author_email='enrico.zorzetto@duke.edu',
   url = 'https://github.com/EnricoZorzetto/mevpy'
   download_url = 'https://github.com/EnricoZorzetto/mevpy/archive/1.01.tar.gz'
   packages=['mevpy'],  #same as name
   install_requires=['matplotlib', 'pandas', 'numpy', 'scipy'], #external packages as dependencies
)


