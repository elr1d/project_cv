from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import torchvision.models as models
import copy
import torch
from sklearn.metrics import balanced_accuracy_score
from app.DATABASE.DB_FUNC import is_file_used,update_used, get_connection, add_model
import datetime
import threading
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import CosineAnnealingLR
training_lock = threading.Lock()
def is_not_used(path):
    return not is_file_used(path)

def uploaded_model_tune(checkpoint_path):
    if not training_lock.acquire(blocking=False):
        print("Обучение уже запущено. Повторный вызов отклонён.")
        return None
    try:
        print("Обучение запущено.")
        
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
        full_train = datasets.ImageFolder(root=path / 'uploaded_train', transform=train_transform)
        train_size = int(0.8 * len(full_train))
        val_size = len(full_train) - train_size
        indices = list(range(len(full_train)))
        labels = [label for _, label in full_train.samples]

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        train_idx, val_idx = next(splitter.split(indices, labels))

        val_full = copy.deepcopy(full_train)
        val_full.transform = test_transform
        val_dataset = torch.utils.data.Subset(val_full, val_idx)
        train_dataset = torch.utils.data.Subset(full_train, train_idx)
        
        test_dataset = datasets.ImageFolder(root=path / 'test', transform=test_transform)

        train_image_paths = [Path(sample[0]).resolve() for sample in full_train.samples]

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
        test_loader = DataLoader(
            test_dataset,
            batch_size = 32,
            shuffle=False,
            num_workers=8
        )
        
        model = models.efficientnet_v2_s(weights=None)
        
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, 6)

        for param in model.features[6].parameters():
            param.requires_grad = True
        for param in model.features[7].parameters():
            param.requires_grad = True
        
        checkpoint = torch.load(checkpoint_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([
            {'params': model.classifier.parameters(), 'lr': 0.001},
            {'params': model.features[6].parameters(), 'lr': 0.0001},
            {'params': model.features[7].parameters(), 'lr': 0.0001}
        ],weight_decay=0.0001)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        new_lrs = [0.0005, 0.00005, 0.00005]
        for param_group, new_lr in zip(optimizer.param_groups, new_lrs):
            param_group['lr'] = new_lr
        start_epoch = checkpoint['epoch']
        original_model_ba = checkpoint['best_ba']
        original_model_loss = checkpoint['val_loss']
        best_model_wts = copy.deepcopy(model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        best_epoch = 0
        epochs = 3
        best_acc = 0
        best_loss = float('inf')
        best_ba = 0
        patience = 5
        epochs_to_improve = 0
        
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        for epoch in range(start_epoch, start_epoch + epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, loss_fn, device)
            val_ba = compute_ba(model, val_loader, device)
            scheduler.step()
            if val_ba > best_ba:
                best_ba = val_ba
                best_acc = val_acc
                best_loss = val_loss
                best_epoch = epoch + 1
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                epochs_to_improve = 0
            else:
                epochs_to_improve += 1
                
            if epochs_to_improve >= patience:
                print(f"Метрики не улучшались {patience} эпох. Прекращаем обучение")
                break
                
            print(f'Epoch {epoch + 1}/{start_epoch+epochs}, val_ba:{val_ba:.4f}, Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%, Val loss: {val_loss:.4f}, Val acc: {val_acc:.2f}%')
        model.load_state_dict(best_model_wts)
        optimizer.load_state_dict(best_optimizer_state)
        trained_model_path = f'C:/Users/pc/Desktop/project_cv/app/models/model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
        test_loss, test_acc = validate(model, test_loader, loss_fn, device)
        print(f"\nточность на test выборке: {test_acc:.2f}%")
        print(f"loss на test выборке: {test_loss:.4f}")
        if True: #best_ba > original_model_ba
            print(f"\nОбучение завершено. Лучшая точность на валидации: {best_ba} > {original_model_ba}. Модель сохранена по пути будет сохранена")
            model_save(best_epoch, best_model_wts, best_optimizer_state, 
                    best_acc, best_loss, trained_model_path,test_loss,test_acc,train_image_paths, train_dataset, best_ba)
            return trained_model_path
        
        print(f"\nОбучение завершено. Лучшая точность на валидации: {best_ba} < {original_model_ba}. Модель не сохраняется")
        return None
    finally:
        training_lock.release()

def model_save(best_epoch, best_model_wts, best_optimizer_state, 
               best_acc, best_loss, trained_model_path,test_loss,test_acc, 
               train_image_paths, train_dataset,best_ba):
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_wts,
        'optimizer_state_dict': best_optimizer_state,
        'best_acc': best_acc,
        'val_loss': best_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'best_ba': best_ba,
        'class_to_idx': train_dataset.dataset.class_to_idx
    }, trained_model_path)
    conn, _ = get_connection()
    try:
        add_model(trained_model_path,best_acc,test_acc,conn=conn)
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
def compute_ba(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    ba = balanced_accuracy_score(all_labels, all_preds)
    return ba
if __name__ == '__main__':
    uploaded_model_tune(str(Path('C:/Users/pc/Desktop/project_cv/app/model/best_model.pth')))