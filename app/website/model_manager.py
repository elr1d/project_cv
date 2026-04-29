import torch 
from torchvision import models, transforms
from PIL import Image
from app.DATABASE.DB_FUNC import get_newest_model_path,get_model_date
class ModelManager:
    def __init__(self):
        self.TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.efficientnet_v2_s(weights = None)
        self.num_classes = 2
        self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, self.num_classes)
        self.model_path = None
        self.checkpoint = None
    def get_model_info(self):
        return f'Модель: efficientnet_v2_s Версия: {get_model_date(self.model_path)} Точность: {self.checkpoint['best_acc']:.2f}%'
    def get_model(self):
        return self.model
    
    def load_last_model(self):
        self.model_path = get_newest_model_path()
        self.checkpoint = torch.load(self.model_path)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self,image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.TRANSFORM(image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device) 
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = predicted_idx.item()
        confidence = confidence.item()
        print(f'Предсказанный класс: {predicted_class}, уверенность: {confidence}')
        return predicted_class, confidence
    
    