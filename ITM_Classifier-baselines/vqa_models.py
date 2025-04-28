import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, RobertaModel, CLIPModel, CLIPProcessor
import timm

class CNNBertClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # ResNet visual encoder
        self.visual_encoder = models.resnet50(pretrained=True)
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        self.visual_encoder.fc = nn.Identity()
        
        # BERT text encoder
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # Trainable fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 768 + 768, 512),  # ResNet + 2 * BERT embeddings
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, image, question_embedding, answer_embedding):
        # Image encoding
        visual_features = self.visual_encoder(image)
        
        # Use pre-computed text embeddings
        combined = torch.cat([visual_features, question_embedding, answer_embedding], dim=1)
        output = self.fusion(combined)
        return output
        
    def train(self, mode=True):
        super().train(mode)
        # Keep encoders in eval mode
        self.visual_encoder.eval()
        self.text_encoder.eval()
        return self

class ViTRoBERTaClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Vision Transformer encoder
        self.visual_encoder = timm.create_model(
            'vit_base_patch16_224', 
            pretrained=True,
            num_classes=0
        )
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        
        # RoBERTa text encoder
        self.text_encoder = RobertaModel.from_pretrained('roberta-base')
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Trainable fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(768 + 768 + 768, 512),  # ViT + 2 * RoBERTa embeddings
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, image, question_embedding, answer_embedding):
        # Image encoding
        visual_features = self.visual_encoder(image)
        
        # Use pre-computed text embeddings
        combined = torch.cat([visual_features, question_embedding, answer_embedding], dim=1)
        output = self.fusion(combined)
        return output
        
    def train(self, mode=True):
        super().train(mode)
        # Keep encoders in eval mode
        self.visual_encoder.eval()
        self.text_encoder.eval()
        return self

class CLIPZeroShotClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze CLIP
        for param in self.clip.parameters():
            param.requires_grad = False
            
    def forward(self, image, question_embedding, answer_embedding):
        # Process inputs
        image_features = self.clip.get_image_features(image)
        text_features = torch.cat([question_embedding, answer_embedding], dim=1)
        
        # Calculate similarity
        similarity = self.clip.logit_scale.exp() * (image_features @ text_features.T)
        
        return similarity
        
    def train(self, mode=True):
        super().train(mode)
        # Always keep CLIP in eval mode
        self.clip.eval()
        return self
        
def get_optimizer_grouped_parameters(model):
    """
    Prepare optimizer with different learning rates for different layers
    """
    no_decay = ['bias', 'LayerNorm.weight']
    
    # Parameters to update
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0
        }
    ]
    
    return optimizer_grouped_parameters