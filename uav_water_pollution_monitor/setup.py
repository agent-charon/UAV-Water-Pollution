from setuptools import setup, find_packages

setup(
    name="uav_water_pollutant_detection",
    version="0.1.0",
    author="Your Name/Team",
    author_email="your.email@example.com",
    description="Implementation of 'UAV-Based Water Pollutants Detection and Classification'",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/uav_water_pollutant_detection", # Replace with your repo URL
    packages=find_packages(exclude=["tests*", "docs*", "data*"]),
    install_requires=[
        "numpy",
        "opencv-python",
        "torch>=1.9.0", # Specify version compatible with your CUDA if using GPU
        "torchvision>=0.10.0",
        "torchaudio>=0.9.0",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "pyyaml",
        "tqdm",
        "ultralytics", # Or specific yolo components
        "timm",
        "pytorch-tabnet>=3.1.1", # Check latest compatible version
        "xgboost",
        "paho-mqtt",
        "Pillow",
        "seaborn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Or your chosen license
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha", # Or appropriate status
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires='>=3.8',
)