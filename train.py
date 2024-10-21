import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LxmertTokenizer, LxmertModel, LxmertConfig
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from PIL import Image
import torchvision.transforms as transforms
import random
from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
import os


class ComparisonDataset(Dataset):
    def __init__(self, json_file, image_dir, tokenizer, max_length=128):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load images
        image_before = Image.open(f"{self.image_dir}/{item['contents'][1]['name']}.png").convert('RGB')
        image_after = Image.open(f"{self.image_dir}/{item['contents'][0]['name']}.png").convert('RGB')
        
        image_before = self.transform(image_before)
        image_after = self.transform(image_after)
        
        # Randomly select an attribute
        attribute = random.choice(item['attributes'])
        
        # Tokenize question
        inputs = self.tokenizer.encode_plus(
            attribute['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create label
        label = 1 if attribute['answer'] == 'After' else 0
        
        return {
            'image_before': image_before,
            'image_after': image_after,
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class CustomLXMERT(torch.nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.lxmert = LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.classifier = torch.nn.Linear(self.lxmert.config.hidden_size, num_labels)

        # Vision model for feature extraction
        self.vision_model = resnet50(pretrained=True)
        self.vision_model = create_feature_extractor(self.vision_model, return_nodes={'avgpool': 'features'})
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, image_before, image_after):
        # Extract features from both images
        with torch.no_grad():
            features_before = self.vision_model(image_before)['features'].squeeze(-1).squeeze(-1)
            features_after = self.vision_model(image_after)['features'].squeeze(-1).squeeze(-1)
        
        # Combine features
        visual_feats = torch.stack([features_before, features_after], dim=1)
        
        # Create visual_pos tensor and convert to float
        visual_pos = torch.zeros(visual_feats.size(0), visual_feats.size(1), 4).to(visual_feats.device)
        visual_pos[:, :, :2] = torch.arange(2).unsqueeze(0).repeat(visual_feats.size(0), 1).unsqueeze(-1).to(visual_feats.device).float()

        outputs = self.lxmert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_feats=visual_feats,
            visual_pos=visual_pos
        )
        pooled_output = outputs.pooled_output
        logits = self.classifier(pooled_output)
        return logits


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, num_epochs, checkpoint_dir='checkpoints', log_dir='logs', start_epoch=0, best_val_acc=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_epoch = start_epoch
        self.best_val_acc = best_val_acc

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image_before = batch['image_before'].to(self.device)
                image_after = batch['image_after'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(input_ids, attention_mask, image_before, image_after)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            train_loss /= len(self.train_loader)
            train_acc = train_correct / train_total

            val_loss, val_acc = self.validate()

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Save checkpoint if validation accuracy improves
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc)

            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        self.writer.close()

    def validate(self):
        self.model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image_before = batch['image_before'].to(self.device)
                image_after = batch['image_after'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(input_ids, attention_mask, image_before, image_after)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = val_correct / val_total

        return val_loss, val_acc

    def save_checkpoint(self, epoch, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}_acc_{val_acc:.4f}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def inference(model, dataloader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_before = batch['image_before'].to(device)
            image_after = batch['image_after'].to(device)

            outputs = model(input_ids, attention_mask, image_before, image_after)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    
    return predictions


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tokenizer and model
    tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
    model = CustomLXMERT().to(device)

    if args.mode == 'train':
        # Create datasets and dataloaders
        train_dataset = ComparisonDataset(args.train_json, args.train_image_dir, tokenizer)
        val_dataset = ComparisonDataset(args.val_json, args.val_image_dir, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        # Initialize optimizer
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

        start_epoch = 0
        best_val_acc = 0

        # Load checkpoint if provided
        if args.resume_checkpoint:
            checkpoint = torch.load(args.resume_checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['val_acc']
            print(f"Resuming training from epoch {start_epoch}")

        # Create trainer
        trainer = Trainer(model, train_loader, val_loader, optimizer, device, 
                          num_epochs=args.num_epochs, 
                          checkpoint_dir=args.checkpoint_dir,
                          log_dir=args.log_dir,
                          start_epoch=start_epoch,
                          best_val_acc=best_val_acc)

        # Start training
        trainer.train()

    elif args.mode == 'inference':
        # Load the trained model
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Create dataset and dataloader for inference
        inference_dataset = ComparisonDataset(args.inference_json, args.inference_image_dir, tokenizer)
        inference_loader = DataLoader(inference_dataset, batch_size=args.batch_size)

        # Run inference
        predictions = inference(model, inference_loader, device)

        # Save predictions
        with open(args.output_file, 'w') as f:
            json.dump(predictions, f)

        print(f"Predictions saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run inference on the LXMERT model")
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True, help='Mode: train or inference')
    parser.add_argument('--train_json', type=str, help='Path to training JSON file')
    parser.add_argument('--train_image_dir', type=str, help='Path to training image directory')
    parser.add_argument('--val_json', type=str, help='Path to validation JSON file')
    parser.add_argument('--val_image_dir', type=str, help='Path to validation image directory')
    parser.add_argument('--inference_json', type=str, help='Path to inference JSON file')
    parser.add_argument('--inference_image_dir', type=str, help='Path to inference image directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--checkpoint_path', type=str, help='Path to load checkpoint for inference')
    parser.add_argument('--output_file', type=str, default='predictions.json', help='Path to save inference predictions')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save TensorBoard logs')
    parser.add_argument('--resume_checkpoint', type=str, help='Path to checkpoint to resume training from')

    args = parser.parse_args()
    main(args)
