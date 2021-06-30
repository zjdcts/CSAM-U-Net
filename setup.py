from setuptools import setup, find_namespace_packages

setup(name='3dunet',
      packages=find_namespace_packages(include=["unet3d", "unet3d.*"]),
      version='1.0.0',
      install_requires=[
            "numpy",
            "scipy",
            "nibabel",
            "nilearn",
            "pandas",
            "h5py"
      ]
      )