import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup
from MIC.model.instructblip import InstructBlipConfig, InstructBlipForConditionalGeneration, InstructBlipProcessor
from PIL import Image
import json
import os
from torch.utils.tensorboard import SummaryWriter


class ImageComparisonDataset(Dataset):
    def __init__(self, root_path: str, img_path: str, json_file: str, processor):
        # Get absolute path
        self.root_path = root_path
        self.img_path = img_path
        json_file = os.path.join(root_path, json_file)

        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self._flattened_data = self._flattened_data()

    def _flattened_data(self):
        flattened_data = []
        for item in self.data:
            attributes = item['attributes']
            for vqa in attributes:
                flattened_data.append({
                    'name': item['name'],
                    'img1': os.path.join(
                        self.root_path,
                        self.img_path,
                        item['contents'][0]['name'] + ".png",
                    ),
                    'img2': os.path.join(
                        self.root_path,
                        self.img_path,
                        item['contents'][1]['name'] + ".png",
                    ),
                    'width': item['contents'][0]['width'],
                    'height': item['contents'][0]['height'],
                    'question': vqa['question'],
                    'answer': vqa['answer']
                })
        return flattened_data

    def __len__(self):
        return len(self._flattened_data)

    def __getitem__(self, idx):
        item = self._flattened_data[idx]
        image1 = Image.open(item['img1']).convert('RGB')
        image2 = Image.open(item['img2']).convert('RGB')

        question = item['question']
        answer = item['answer']

        inputs = self.processor(
            images=[image1, image2],
            text=question,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Remove batch dimension
        for k, v in inputs.items():
            inputs[k] = v.squeeze()


        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        inputs['img_mask'] = torch.tensor([[1 for i in range(2)]])
        # inputs['img_mask'] = torch.tensor([1, 1])
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs['labels'] = self.processor.tokenizer(
                answer,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze()
        return inputs


# Fine-tuning function (updated)
def fine_tune_instructblip(
        model,
        train_dataloader,
        val_dataloader,
        num_epochs,
        device,
        output_dir
):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10,
                                                num_training_steps=total_steps)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                pixel_values=batch['pixel_values'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                img_mask=batch['img_mask'],
                do_sample=False,
                max_length=50,
                min_length=1,
                set_min_padding_size=False,
                labels=batch['labels']
            )
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average train loss: {avg_train_loss:.4f}")
        writer.add_scalar('train/epoch_loss', avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss:.4f}")
        writer.add_scalar('val/loss', avg_val_loss, epoch)

        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")

    writer.close()
    return model


# Main execution (updated)
def main():
    model_ckpt = "BleachNick/MMICL-Instructblip-T5-xl"
    processor_ckpt = "Salesforce/instructblip-flan-t5-xl"

    config = InstructBlipConfig.from_pretrained(model_ckpt)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        model_ckpt,
        config=config
    ).to('cuda:0',dtype=torch.bfloat16)
    processor = InstructBlipProcessor.from_pretrained(processor_ckpt)

    # Add custom tokens if necessary
    image_placeholder = "图"
    sp = [image_placeholder] + [f"<image{i}>" for i in range(20)]
    sp = sp + processor.tokenizer.additional_special_tokens[len(sp):]
    processor.tokenizer.add_special_tokens({'additional_special_tokens': sp})

    if model.qformer.embeddings.word_embeddings.weight.shape[0] != len(processor.qformer_tokenizer):
        model.qformer.resize_token_embeddings(len(processor.qformer_tokenizer))

    # Prepare dataset and dataloader
    train_dataset = ImageComparisonDataset(
        root_path='data',
        img_path='train',
        json_file='class_train_annotation.json',
        processor=processor
    )
    test_dataset = ImageComparisonDataset(
        root_path='data',
        img_path='val',
        json_file='class_val_annotation.json',
        processor=processor
    )

    # Split the dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set up output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Fine-tune the model
    num_epochs = 3
    fine_tuned_model = fine_tune_instructblip(model, train_dataloader, val_dataloader, num_epochs, device, output_dir)

    # Save the final fine-tuned model and processor
    fine_tuned_model.save_pretrained(os.path.join(output_dir, "final_model"))
    processor.save_pretrained(os.path.join(output_dir, "final_processor"))


if __name__ == "__main__":
    main()
