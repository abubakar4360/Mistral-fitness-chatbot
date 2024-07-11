import pandas as pd
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_from_disk
from trl import SFTTrainer

from preprocessing.Data_preprocessing import clean_data, structure_data
from utils.huggingface_login import huggingface_login


def main(file_path, nrows, model_name, device):
    file = pd.read_excel(file_path, nrows=nrows)
    file = file[['Post Payment Form Data', 'Check-in Data', 'Final Feedback']]

    # Cleaning data and structuring data
    text = file.applymap(clean_data)
    dataset = structure_data(text)

    # Huggingface login and access tokens
    huggingface_login()

    # Quantization
    compute_dtype = torch.float16
    use_4bit = True

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,
        use_flash_attention_2=True,
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    # model.gradient_checkpointing_enable()

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    lora_target_modules = ["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"]
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # Set training parameters
    training_arguments = TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        logging_steps=1,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        num_train_epochs=5,
        evaluation_strategy="steps",
        eval_steps=0.1,
        warmup_ratio=0.05,
        save_strategy="epoch",
        group_by_length=True,
        output_dir="weights_500",
        save_safetensors=True,
        lr_scheduler_type="cosine",
        seed=42,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['Train'],
        eval_dataset=dataset["Validation"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )
    # Train model
    trainer.train()

    # Save trained model
    trainer.save_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--upload_file', required=True, help='Path to excel file')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows from excel file')

    args = parser.parse_args()

    file_path = args.upload_file
    nrows = args.nrows

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    main(file_path, nrows, model_name, device)