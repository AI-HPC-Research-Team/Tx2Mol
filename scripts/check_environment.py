"""Check the supported environment without allocating a model or changing files."""
import importlib.metadata
import json
import platform


def main():
    import torch
    from flash_attn.models.gpt import GPTLMHeadModel  # noqa: F401
    from transformers import AutoTokenizer  # noqa: F401
    from rdkit import Chem  # noqa: F401

    packages = ["torch", "transformers", "accelerate", "rdkit", "flash-attn", "numpy"]
    info = {"python": platform.python_version(), "packages": {
        name: importlib.metadata.version(name) for name in packages}}
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["compute_capability"] = list(torch.cuda.get_device_capability(0))
        if info["compute_capability"][0] < 8:
            raise RuntimeError("Tx2Mol's archived FlashAttention-2 model requires an Ampere or newer NVIDIA GPU. GeneVAE alone supports CPU/V100.")
    else:
        raise RuntimeError("A CUDA GPU is required for the archived Tx2Mol model; GeneVAE alone can run on CPU.")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
