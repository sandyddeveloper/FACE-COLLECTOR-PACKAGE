# How to use the Face Collector CLI

To enable the `face-collector` command on your system, you need to install the package in editable mode. This allows you to run the collector from any directory and ensures that any changes you make to the code are immediately reflected in the command.

## ⚙️ Installation

Open your terminal in the project root (`c:\Users\SANTHU\OneDrive\Desktop\CCTV-FINAL-OUTPUT\FACE-COLLECTOR-PACKAGE`) and run:

```bash
pip install -e .
```

## 🚀 Usage

Once installed, you can use the `face-collector` command directly:

### Basic Example
```bash
face-collector --stream-url="http://192.168.1.50:8080/video" --output-dir="./captures"
```

### With custom API URL
```bash
face-collector --api-url="http://your-server.com/api" --camera-id="10"
```

### Full Metadata Support
```bash
face-collector --camera-id="1" --org-id="7" --device-name="CCTV-01"
```

## 📋 Available Arguments
- `--stream-url`: Video stream source (IP camera URL)
- `--api-url`: Backend API endpoint
- `--output-dir`: Base directory (though images are no longer saved to disk by default)
- `--camera-id`, `--device-id`, `--device-name`, `--org-id`: API metadata
