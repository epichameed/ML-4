import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import pickle
import time
from tqdm import tqdm
import wandb
from vqa_models import CNNBertClassifier, ViTRoBERTaClassifier, CLIPZeroShotClassifier

class VQADataset(Dataset):
    def __init__(self, images_path, data_file, sentence_embeddings, split='train', train_ratio=1.0):
        self.images_path = images_path
        self.split = split
        self.train_ratio = train_ratio if split == "train" else 1.0
        self.sentence_embeddings = sentence_embeddings
        
        # Image data
        self.image_paths = []
        self.questions = []
        self.answers = []
        self.labels = []
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Load data
        self.load_data(data_file)
        
    def load_data(self, data_file):
        print(f"Loading {self.split} data from {data_file}")
        
        with open(data_file, 'r') as f:
            lines = f.readlines()
            
            if self.split == "train":
                num_samples = int(len(lines) * self.train_ratio)
                lines = lines[:num_samples]
            
            for line in lines:
                img_name, text, label = line.strip().split('\t')
                img_path = os.path.join(self.images_path, img_name.strip())
                
                # Skip if image doesn't exist
                if not os.path.exists(img_path):
                    continue
                
                # Split question and answer
                qa = text.split('?')
                question = qa[0].strip() + '?'
                answer = qa[1].strip()
                
                self.image_paths.append(img_path)
                self.questions.append(question)
                self.answers.append(answer)
                self.labels.append(1 if label == "match" else 0)
                
        print(f"Loaded {len(self.image_paths)} samples")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        try:
            # Load and preprocess image
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image = self.transform(image)
            
            # Get question and answer embeddings
            question_embedding = torch.tensor(
                self.sentence_embeddings[self.questions[idx]], 
                dtype=torch.float32
            )
            answer_embedding = torch.tensor(
                self.sentence_embeddings[self.answers[idx]], 
                dtype=torch.float32
            )
            
            # Get label
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            
            return {
                'image': image,
                'question_embedding': question_embedding,
                'answer_embedding': answer_embedding,
                'label': label
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            # Return a dummy sample
            return {
                'image': torch.zeros((3, 224, 224)),
                'question_embedding': torch.zeros(768),
                'answer_embedding': torch.zeros(768),
                'label': torch.tensor(0)
            }

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        try:
            # Move data to device
            images = batch['image'].to(device)
            question_embeddings = batch['question_embedding'].to(device)
            answer_embeddings = batch['answer_embedding'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, question_embeddings, answer_embeddings)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Clear cache and skip batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    
    return total_loss / len(dataloader), 100. * correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating')
        for batch in progress_bar:
            try:
                images = batch['image'].to(device)
                question_embeddings = batch['question_embedding'].to(device)
                answer_embeddings = batch['answer_embedding'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, question_embeddings, answer_embeddings)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    return total_loss / len(dataloader), 100. * correct / total

def main():
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/gdrive')
        base_path = "/content/ML-4/ITM_Classifier-baselines"
    except ImportError:
        base_path = "."
    
    # Parameters
    BATCH_SIZE = 8  # Reduced batch size for Colab
    NUM_EPOCHS = 30
    LEARNING_RATE = 2e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    IMAGES_PATH = os.path.join(base_path, "visual7w-images")
    TRAIN_FILE = os.path.join(base_path, "visual7w-text/v7w.TrainImages.itm.txt")
    DEV_FILE = os.path.join(base_path, "visual7w-text/v7w.DevImages.itm.txt")
    TEST_FILE = os.path.join(base_path, "visual7w-text/v7w.TestImages.itm.txt")
    EMBEDDINGS_FILE = os.path.join(base_path, "v7w.sentence_embeddings-gtr-t5-large.pkl")
    
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load sentence embeddings
    print("Loading sentence embeddings...")
    with open(EMBEDDINGS_FILE, 'rb') as f:
        sentence_embeddings = pickle.load(f)
    
    # Create datasets with smaller training subset
    train_dataset = VQADataset(
        IMAGES_PATH, 
        TRAIN_FILE,
        sentence_embeddings,
        'train',
        train_ratio=0.1  # Using 10% of training data for Colab
    )
    
    dev_dataset = VQADataset(
        IMAGES_PATH, 
        DEV_FILE,
        sentence_embeddings,
        'dev'
    )
    
    test_dataset = VQADataset(
        IMAGES_PATH, 
        TEST_FILE,
        sentence_embeddings,
        'test'
    )
    
    # Create dataloaders with appropriate settings for Colab
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Initialize wandb
    wandb.init(project="vqa-task", config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": NUM_EPOCHS,
        "device": str(DEVICE)
    })
    
    # Train and evaluate models
    models = {
        'cnn-bert': CNNBertClassifier(),
        # Commenting out heavier models for Colab
        # 'vit-roberta': ViTRoBERTaClassifier(),
        # 'clip': CLIPZeroShotClassifier()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        model = model.to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        
        best_dev_acc = 0
        best_epoch = 0
        
        try:
            for epoch in range(NUM_EPOCHS):
                print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
                
                # Train
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, DEVICE
                )
                
                # Evaluate on dev set
                dev_loss, dev_acc = evaluate(
                    model, dev_loader, criterion, DEVICE
                )
                
                # Update learning rate
                scheduler.step()
                
                # Log metrics
                wandb.log({
                    f"{name}_train_loss": train_loss,
                    f"{name}_train_acc": train_acc,
                    f"{name}_dev_loss": dev_loss,
                    f"{name}_dev_acc": dev_acc,
                    "epoch": epoch
                })
                
                # Save best model
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_epoch = epoch
                    save_path = os.path.join("/content/gdrive/MyDrive/ML4_models", f'best_{name}_model.pth')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(model.state_dict(), save_path)
                    
        except KeyboardInterrupt:
            print("Training interrupted by user")
        
        # Load best model and evaluate on test set
        try:
            model_path = os.path.join("/content/gdrive/MyDrive/ML4_models", f'best_{name}_model.pth')
            model.load_state_dict(torch.load(model_path))
            test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
            
            results[name] = {
                'test_acc': test_acc,
                'best_dev_acc': best_dev_acc,
                'best_epoch': best_epoch
            }
            
            print(f"\nFinal results for {name}:")
            print(f"Best Dev Accuracy: {best_dev_acc:.2f}% (Epoch {best_epoch+1})")
            print(f"Test Accuracy: {test_acc:.2f}%")
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Log final results
    for name, metrics in results.items():
        wandb.run.summary[f"{name}_test_acc"] = metrics['test_acc']
        wandb.run.summary[f"{name}_best_dev_acc"] = metrics['best_dev_acc']
    
    wandb.finish()

if __name__ == '__main__':
    main()