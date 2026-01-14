import os
import torch
import warnings
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    logging,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from liger_kernel.transformers import apply_liger_kernel_to_qwen2

# 0. Configuration and hyperparameters
# ==============================================================================
MODEL_NAME_OR_PATH = "./base_model/Qwen2.5-Coder-7B-Instruct"
DATASET_PATH = "final_dataset/"
OUTPUT_DIR = "gcode-generator-Qwen1-G-Coder"
WANDB_PROJECT = "gcode-generator"

# Training parameters
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 5
MAX_SEQ_LENGTH = 18000
LOGGING_STEPS = 10
SAVE_STEPS = 100
WARMUP_STEPS = 30


# 1. Define the list of Tokens that need to be expanded
# ==============================================================================
def get_tokens_to_add():
    standard_g_codes = [
        "G0", "G1", "G2", "G3", "G4", "G10", "G11", "G20", "G21", "G28", "G29", "G90", "G91", "G92",
        " X", " Y", " E", " F", "Z.2"
    ]
    standard_m_codes = [
        "M0", "M1", "M17", "M73", "M82", "M83", "M84", "M104", "M105", "M106", "M107",
        "M109", "M140", "M190", "M204", "M220", "M221", "M400", "M600"
    ]
    bambu_custom_codes = [
        "G29.1", "G29.2", "M412", "M622", "M900", "M1002",
        "; FEATURE:", "; LINE_WIDTH:", "; LAYER:",
        "M624 AQAAAAAAAAA="
    ]
    common_feedrates = [
        " F1800", " F1200", " F42000", " F3600"
    ]
    prompt_keywords_tokens = [
        "module=",
        " teeth_count=",
        " bore_diameter="
    ]
    long_comment_prefixes = [
        "; start printing object, unique label id: 15",
        "; stop printing object, unique label id: 15"
    ]

    return standard_g_codes + standard_m_codes + bambu_custom_codes + \
        common_feedrates + prompt_keywords_tokens + long_comment_prefixes


# 2. Load the native Tokenizer and extend it
# ==============================================================================
def load_base_tokenizer_and_extend(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )

    special_tokens_dict = {
        "additional_special_tokens": ["<|endofprompt|>"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    tokens_to_add = get_tokens_to_add()
    existing_vocab = tokenizer.get_vocab()
    new_tokens = [t for t in tokens_to_add if t not in existing_vocab]

    if new_tokens:
        num_added = tokenizer.add_tokens(new_tokens)
        print(f"The vocabulary has been expanded, with the addition of G-code related tokens: {num_added}")

    print(f"Tokenizer loaded successfully, current vocabulary size: {len(tokenizer)}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"
    return tokenizer


# 3. Data preprocessing
# ==============================================================================
def tokenize_and_mask_prompt(examples, tokenizer):
    eop_token_id = tokenizer.convert_tokens_to_ids("<|endofprompt|>")
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=MAX_SEQ_LENGTH,
        return_tensors=None,
    )
    labels = [list(ids) for ids in outputs["input_ids"]]
    for i in range(len(labels)):
        try:
            if eop_token_id in labels[i]:
                eop_index = labels[i].index(eop_token_id)
                for j in range(eop_index + 1):
                    labels[i][j] = -100
        except ValueError:
            pass

    outputs["labels"] = labels
    return outputs


def main():
    logging.set_verbosity_info()
    warnings.filterwarnings("ignore")

    # --- Load tokenizer ---
    tokenizer = load_base_tokenizer_and_extend(MODEL_NAME_OR_PATH)

    # --- Load dataset ---
    raw_datasets = load_dataset("json", data_dir=DATASET_PATH, cache_dir=f"{OUTPUT_DIR}/.cache")

    tokenized_datasets = raw_datasets.map(
        tokenize_and_mask_prompt,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=os.cpu_count() // 2,
        remove_columns=["text"],
        desc="Tokenizing dataset",
        load_from_cache_file=False
    )

    # --- Load the model configuration and make modifications ---
    config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    config.max_position_embeddings = MAX_SEQ_LENGTH
    print(f"Modified max_position_embeddings to: {config.max_position_embeddings}")

    # RoPE Scaling
    if MAX_SEQ_LENGTH > 32768:
        config.rope_scaling = {"type": "dynamic", "factor": 2.0}

    # --- Load LLM ---
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    print("Adjusting the size of the Embedding layer...")
    model.resize_token_embeddings(len(tokenizer))

    # --- Smart Initialization ---
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight

        init_map = {
            " X": "X", " Y": "Y", " E": "E", " F": "F", "Z.2": "Z",
            " F1800": "F", " F1200": "F", " F42000": "F", " F3600": "F",
            "module=": "module", " teeth_count=": "teeth",
            " bore_diameter=": "diameter",
            "M624 AQAAAAAAAAA=": "M",
            "<|endofprompt|>": "\n",
        }

        for new_token, ref_token in init_map.items():
            new_id = tokenizer.convert_tokens_to_ids(new_token)
            ref_id = tokenizer.convert_tokens_to_ids(ref_token)
            if new_id is not None and ref_id != tokenizer.unk_token_id:
                embeddings[new_id] = embeddings[ref_id].clone()

        tokens_to_add = get_tokens_to_add()
        g_ref_id = tokenizer.convert_tokens_to_ids("G")
        m_ref_id = tokenizer.convert_tokens_to_ids("M")
        comment_ref_id = tokenizer.convert_tokens_to_ids(";")

        count_batch = 0
        for t in tokens_to_add:
            if t in init_map: continue
            t_id = tokenizer.convert_tokens_to_ids(t)
            if t.startswith("G") and g_ref_id != tokenizer.unk_token_id:
                embeddings[t_id] = embeddings[g_ref_id].clone()
                count_batch += 1
            elif t.startswith("M") and m_ref_id != tokenizer.unk_token_id:
                embeddings[t_id] = embeddings[m_ref_id].clone()
                count_batch += 1
            elif t.startswith(";") and comment_ref_id != tokenizer.unk_token_id:
                embeddings[t_id] = embeddings[comment_ref_id].clone()
                count_batch += 1

    print(f"Smart Initialization completed! Precise mapping of {len(init_map)} items, batch mapping of {count_batch} items.")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # --- LoRA config ---
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Training arguments config ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        eval_strategy="no",

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=False,

        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        report_to="wandb",
        run_name=f"{WANDB_PROJECT}-Qwen1-G-Coder",

        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Start fine-tuning ---
    if list(path for path in os.listdir(training_args.output_dir) if "checkpoint" in path):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # --- Save fine-tuned LLM ---
    final_path = f"{OUTPUT_DIR}/final_model"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)


if __name__ == "__main__":
    apply_liger_kernel_to_qwen2()
    main()