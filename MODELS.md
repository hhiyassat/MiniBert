# Model Files

The trained model files are too large (>1GB each) to be stored directly in GitHub. 

## Available Models

| Model Folder | Size | Best Loss | Download |
|--------------|------|-----------|----------|
| `model/` | 1.1GB | 5.7423 | [Download Link] |
| `model_750K_ver1_edition/` | 1.1GB | 5.2692 | [Download Link] |
| `model_750K_ver2_edition/` | 1.1GB | 5.2692 | [Download Link] |
| `model 1000000 ver1_edition/` | 1.1GB | 5.7423 | [Download Link] |
| `model_500000/` | 348MB | 8.2942 | [Download Link] |
| `model_eng/` | 215MB | 3.5428 | [Download Link] |

## Using Git LFS (Recommended)

To push large model files to GitHub, you need to use Git Large File Storage (LFS):

### Installation

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Or download from: https://git-lfs.github.com/
```

### Setup

```bash
# Initialize Git LFS
git lfs install

# Track .pt files
git lfs track "*.pt"

# Add and commit
git add .gitattributes
git add model/*.pt
git commit -m "Add model files via Git LFS"
git push origin main
```

## Alternative: External Storage

You can also host models on:
- **Hugging Face Hub**: https://huggingface.co/
- **Google Drive**: Share download links
- **Cloud Storage**: AWS S3, Google Cloud Storage, etc.

Then update this file with download links.

