# Possible implementation toward Emoji-Kitchen
This is an repo for the possible implementation of [Emoji-Kitchen](https://emojikitchen.dev/) by Google. I will continue try different kinds of methods to achieve the result like Emoji-Kitchen using diffusion model. All suggestion are welcome!! The repo is 
![result](image/001.png)

## Method one: Inpainting + LORA
### 1. Train LoRAs for reference image
In this step, LoRA for one image is trained based on SDXL. Using SDXL here is important because they found that the pre-trained SDXL exhibits strong learning when fine-tuned on only one reference style image.

Fortunately, diffusers already implemented LoRA based on SDXL [here](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md) and you can simply follow the instruction. 

For example, your training script would be like this.
```bash
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# for subject
export OUTPUT_DIR="lora-sdxl-dog"
export INSTANCE_DIR="dog"
export PROMPT="a sbu dog"
export VALID_PROMPT="a sbu dog in a bucket"

# for style
# export OUTPUT_DIR="lora-sdxl-waterpainting"
# export INSTANCE_DIR="waterpainting"
# export PROMPT="a cat of in szn style"
# export VALID_PROMPT="a man in szn style"

accelerate launch train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${PROMPT}" \
  --rank=64 \
  --resolution=1024 \
  --train_batch_size=1 \
  --learning_rate=5e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --validation_prompt="${VALID_PROMPT}" \
  --validation_epochs=50 \
  --seed="0" \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --use_8bit_adam \
  --push_to_hub \
```

### 2. Inpainting using LoRA
Can just follow the code in **inference_inpainting.ipynb** they are based on diffuser and are very easy to understand.
![result](image/002.png)
![result](image/004.png)

### 3. Comparsion between methods
![img](image/003.png)
## TODO

## Method Two: Naive Interpolation
<div style="display: flex; justify-content:  space-between; gap: 10px;">
  <img src="image/out_0_1.gif" alt="GIF 1 description" width="300" height="300"> 
  <img src="image/out_best2.gif" alt="GIF 2 description" width="300" height="300">
</div>

## Method Three: Interpolation plus extra control in the loss function
$L = \mathbb{E}_{x_0,\epsilon\sim\mathcal{N}(0,I),c,t}\left[\|\epsilon - \epsilon_{\theta}(x_t,c,t)\|^2\right] + \|f(x_{0a}) - Interpolation(f(x_{0a1},f(x_{0a2})))\|^2 $

![result](image/005.png)
<div style="display: flex; justify-content:  space-between; gap: 10px;">
  <img src="image/out_0_0.gif" alt="GIF 1 description" width="300" height="300">
</div>
- [x] Some Comparision with differen methods
- [ ] Modified of Method Three