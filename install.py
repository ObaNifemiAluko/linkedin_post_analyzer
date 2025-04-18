import os
import subprocess
import sys

def install_latest_gradio():
    """Install the latest version of Gradio."""
    print("Installing latest Gradio version...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gradio>=4.44.1"])
    print("Successfully installed latest Gradio version.")
    
    # Verify installation
    import gradio
    print(f"Gradio version: {gradio.__version__}")

if __name__ == "__main__":
    install_latest_gradio() 