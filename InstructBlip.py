from MIC.model.instructblip import InstructBlipConfig,InstructBlipForConditionalGeneration,InstructBlipProcessor
from PIL import Image
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
from peft import LoraConfig, get_peft_model

model_type="instructblip"
model_ckpt="BleachNick/MMICL-Instructblip-T5-xl"
processor_ckpt = "Salesforce/instructblip-flan-t5-xl"
config = InstructBlipConfig.from_pretrained(model_ckpt)

def prepare_model_for_lora(model):
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

# image_palceholder="图"
# sp = [image_palceholder]+[f"<image{i}>" for i in range(20)]
# processor = InstructBlipProcessor.from_pretrained(
#     processor_ckpt
# )
# sp = sp+processor.tokenizer.additional_special_tokens[len(sp):]
# processor.tokenizer.add_special_tokens({'additional_special_tokens':sp})
# if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
#     model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))
# replace_token="".join(32*[image_palceholder])


# image = Image.open ("data/val/v1_1_before.png")
# image1 = Image.open ("data/val/v1_1_after.png")
# images = [image,image1]

# prompt = [f'Use the image before: <image0>{replace_token},image after: <image1>{replace_token}'
#           f'The image before is the on the left and the image after is on the right.'
#           f'Given the image before and image after as visual aid to answer the following question:'
#           f'The hem of the vestment in image 1 is softer than in image 2. Which is image 2: After or Before?']
# prompt = " ".join(prompt)

# inputs = processor(images=images, text=prompt, return_tensors="pt")

# inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
# inputs['img_mask'] = torch.tensor([[1 for i in range(len(images))]])
# inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)

# inputs = inputs.to('cuda:0')
# outputs = model.generate(
#         pixel_values = inputs['pixel_values'],
#         input_ids = inputs['input_ids'],
#         attention_mask = inputs['attention_mask'],
#         img_mask = inputs['img_mask'],
#         do_sample=False,
#         max_length=50,
#         min_length=1,
#         set_min_padding_size =False,
# )
# generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
# print(generated_text)

class CustomDataset(Dataset):
    def __init__(self, json_file, processor, image_dir):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.image_dir = image_dir
        self.replace_token = "".join(32 * ["<image_placeholder>"])  # Use a fixed placeholder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        before_image = Image.open(f"{self.image_dir}/{item['contents'][1]['name']}.png")
        after_image = Image.open(f"{self.image_dir}/{item['contents'][0]['name']}.png")
        images = [before_image, after_image]

        attribute = item['attributes'][0]  # You can modify this to use multiple attributes if needed
        prompt = f"Use the image before: <image0>{self.replace_token}, image after: <image1>{self.replace_token} " \
                 f"The image before is on the left and the image after is on the right. " \
                 f"Given the image before and image after as visual aid to answer the following question: " \
                 f"{attribute['question']}"

        inputs = self.processor(images=images, text=prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=256)
        
        # Ensure that the image_embeds are correctly set
        inputs['image_embeds'] = inputs.pop('pixel_values')
        inputs['image_attention_mask'] = torch.ones(inputs['image_embeds'].shape[:-1], dtype=torch.long)

        # Encode the answer and pad it
        labels = self.processor.tokenizer.encode(
            attribute['answer'],
            return_tensors="pt",
            padding='max_length',
            max_length=50,
            truncation=True
        )[0]

        # Ensure labels tensor is always of size 50
        if len(labels) < 50:
            labels = torch.cat([labels, torch.zeros(50 - len(labels), dtype=torch.long)])
        else:
            labels = labels[:50]

        inputs['labels'] = labels

        # Remove batch dimension added by the processor
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 1:
                inputs[key] = inputs[key].squeeze(0)

        return inputs

def train_model(model, processor, train_json, image_dir, num_epochs, batch_size, learning_rate):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    train_dataset = CustomDataset(train_json, processor, image_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Remove the scaler as we'll use regular training without mixed precision
    # scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            inputs['image_embeds'] = inputs['image_embeds'].to(torch.float32)  # Change to float32
            
            inputs.pop('pixel_values', None)
            inputs.pop('img_mask', None)

            inputs['labels'] = inputs['labels'].to(device)

            # Remove torch.cuda.empty_cache() as it's not necessary here

            # Reduce accumulation steps
            accumulation_steps = 2

            # Remove autocast context
            outputs = model(**inputs)
            loss = outputs.loss / accumulation_steps

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    print("Training completed!")

# Modify the model initialization to use float32 instead of bfloat16
model = InstructBlipForConditionalGeneration.from_pretrained(
    model_ckpt,
    config=config).to('cuda:0', dtype=torch.float32)
model = prepare_model_for_lora(model)

# Reduce batch size
train_model(model, processor, train_json, image_dir, num_epochs=5, batch_size=1, learning_rate=1e-5)
