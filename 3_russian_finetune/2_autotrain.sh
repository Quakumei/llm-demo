# /bin/sh
autotrain llm \
    --train \
    --project_name mistral-7b-mj-finetuned \
    --model Open-Orca/Mistral-7B-OpenOrca \
    --data_path . \
    --use_peft --use_int4 \
    --learning_rate 2e-4 \
    --train_batch_size 12 \
    --num_train_epochs 45 \
    --trainer sft \
    --target_modules q_proj,v_proj \
    --push_to_hub --repo_id quakumei/Mistral-OpenOrca-ft-45it

