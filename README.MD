# Personality-Finetuning for Language Models

Scripts to give an LLM a personality by using Supervised Finetuning (SFT) with a small dataset of ~ 1000 examples. By training the LLM with example chats in the style of a character, we can get it to have a bit more edge than your traditional LLM offering.

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone [URL-of-this-repo]
   ```
2. Download the dataset file provided or create your own dataset tailored to the personality you aim to model.

## Usage

1. Run the finetuning script:
   ```bash
   python finetune_mistral.py
   ```
2. Set the following variables in the script before running:
   - `model_path`: Path to the downloaded LLM, e.g., Mistral7b.
   - `dataset_path`: Path to the dataset file.
   - `output_dir`: Path where you want the LoRA adapter to be saved.

3. Adjust important parameters based on your hardware's capabilities:
   - `max_seq_length`: Maximum sequence length for examples; higher values require more hardware resources.
   - `per_device_train_batch_size`: Adjust based on the GPU size and availability (increase for larger/multiple GPUs, decrease for smaller GPUs).

## Merging the Adapter

After training completes, merge the LoRA adapter with the original model:
```bash
python merge_peft.py [--base_model_name_or_path BASE_MODEL_NAME_OR_PATH] [--peft_model_path PEFT_MODEL_PATH] [--output_dir OUTPUT_DIR]
```

## Using the Finetuned Model

To convert the finetuned model for use with Ollama:
1. Follow the guide provided by Ollama on importing models:
   - [Ollama Model Import Guide](https://github.com/ollama/ollama/blob/main/docs/import.md)
   - Begin at the "Convert Model" section.

## Multi-GPU Training

For training with multiple GPUs, use the Accelerate library with the provided Distributed Data Parallel (DDP) script:
```bash
accelerate config
accelerate finetune_mistral_ddp.py
```
For further details on Accelerate configuration, consult the [Hugging Face Accelerate documentation](https://huggingface.co/docs/trl/main/en/customization).

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

Thanks to Mistral for open sourcing the model, and Huggingface for great training libraries and documentation.
