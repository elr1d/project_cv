from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import torchvision.models as models
import copy
import torch
from app.DATABASE.DB_FUNC import is_file_used,update_used, get_connection, add_model
import datetime
import threading

training_lock = threading.Lock()
def is_not_used(path):
    return not is_file_used(path)

def uploaded_model_tune(checkpoint_path):
    if not training_lock.acquire(blocking=False):
        print("Обучение уже запущено. Повторный вызов отклонён.")
        return None
    try:
        print("Обучение запущено.")
        model = models.efficientnet_v2_s(weights=None)
        IMG_SIZE = 224
        train_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        path = Path('C:/Users/pc/Desktop/project_cv/app/data').resolve()
        #print("Путь к uploaded_train:", path / 'uploaded_train')
        train_dataset = datasets.ImageFolder(
            root=path / 'uploaded_train',
            transform=train_transform,
            is_valid_file=is_not_used
        )
        train_image_paths = [Path(sample[0]).resolve() for sample in train_dataset.samples]
        val_dataset = datasets.ImageFolder(
            root=path / 'test_set',
            transform=test_transform
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size = 32,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size = 32,
            shuffle=False,
            num_workers=0
        )
        for param in model.parameters():
            param.requires_grad = False
        
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 2)
        checkpoint = torch.load(checkpoint_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005
        start_epoch = checkpoint['epoch']
        original_model_acc = checkpoint['best_acc']
        original_model_loss = checkpoint['val_loss']
        best_model_wts = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        best_epoch = 0
        epochs = 3
        best_acc = 0.0
        best_loss = float('inf')
        for epoch in range(start_epoch, start_epoch+ epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, loss_fn, device)

            if val_acc > best_acc:
                best_acc = val_acc
                best_loss = val_loss
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            print(f'Epoch {epoch + 1}/{start_epoch + epochs}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%, Val loss: {val_loss:.4f}, Val acc: {val_acc:.2f}%')

        model.load_state_dict(best_model_wts)
        optimizer.load_state_dict(best_optimizer_state)
        trained_model_path = f'C:/Users/pc/Desktop/project_cv/app/models/model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        if True: #best_acc > original_model_acc
            model_save(best_epoch, best_model_wts, best_optimizer_state, 
                    best_acc, best_loss, trained_model_path, train_image_paths)
            print(f"\nОбучение завершено. Лучшая точность на валидации: {best_acc} > {original_model_acc}. Модель сохранена по пути будет сохранена")
            return trained_model_path
        
        print(f"\nОбучение завершено. Лучшая точность на валидации: {best_acc} < {original_model_acc}. Модель не сохраняется")
        return None
    finally:
        training_lock.release()

def model_save(best_epoch, best_model_wts, best_optimizer_state, best_acc, best_loss, trained_model_path, train_image_paths):
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_wts,
        'optimizer_state_dict': best_optimizer_state,
        'best_acc': best_acc,
        'val_loss': best_loss
    }, trained_model_path)
    conn, _ = get_connection()
    try:
        add_model(trained_model_path,best_acc,conn=conn)
        for path in train_image_paths:
            update_used(str(path),1, conn=conn)
        conn.commit()
    finally:
        conn.close()
    
#функция обучения
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    
    run_loss = 0.0
    corrects = 0
    total = 0
    
    for images, labels in loader:
        images,labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        run_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        corrects += (predicted == labels).sum().item()

    return run_loss / len(loader), 100 * corrects / total
#функция валидации
def validate(model, loader, loss_fn, device):
    model.eval()

    run_loss = 0.0
    corrects = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images,labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            run_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            corrects += (predicted == labels).sum().item()

    return run_loss / len(loader), 100 * corrects / total

if __name__ == '__main__':
    uploaded_model_tune(str(Path('C:/Users/pc/Desktop/project_cv/app/model/best_model.pth')))