from setuptools import setup

setup(name='anndata_oom',
      version=0.1,
      description='Out of memory tricks for adata',
      url='http://github.com/redst4r/anndata_oom/',
      author='redst4r',
      maintainer='redst4r',
      maintainer_email='redst4r@web.de',
      license='BSD 2-Clause License',
      keywords='scanpy',
      packages=['anndata_oom'],
      install_requires=[
          'scipy',
          'anndata',
          'h5py',
          'tqdm',
          ],
      zip_safe=False)
