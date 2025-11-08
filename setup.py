from setuptools import setup, find_packages

setup(
    name="dustbin-simulator",
    version="1.0.0",
    author="Vijay Shinde",
    description="AI-powered Smart Dustbin Lock Simulator using PyTorch and Streamlit",
    packages=find_packages(),
    install_requires=["torch", "torchvision", "streamlit", "Pillow", "matplotlib"],
    entry_points={"console_scripts": ["dustbin-simulator = dustbin_simulator.app:main"]},
    python_requires=">=3.9",
)
