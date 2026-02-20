# Face Collector

This is a Python package that connects to a video stream, detects faces, aligns them, checks for quality (blur, pose), and saves/sends high-quality face crops.

## Installation from GitHub

You can install this package directly from your GitHub repository using pip once you push the code online:

```bash
pip install git+https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

## Usage

Once installed, you can run the tool from your command line:

```bash
face-collector --stream-url="http://YOUR_STREAM_URL:8080/video" --output-dir="my_faces"
```

If you don't provide arguments, it defaults to `http://192.168.68.103:8080/video` and an `output` folder in the current directory.

## Development (Local Installation)

To install the package locally while you make changes:

```bash
pip install -e .
```
