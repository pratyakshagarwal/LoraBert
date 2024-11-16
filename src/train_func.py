import torch
import torch.nn as nn
from tqdm import tqdm

def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def train(train_loader, test_dl, net, epochs=5, total_iterations_limit=None, device='cpu', lr: float=1e-3):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, total_steps=epochs * len(train_loader))
    metrics_fn = calculate_accuracy
    total_iterations = 0

    net.to(device)
    for epoch in range(epochs):
        net.train()

        loss_sum, acc_sum = 0., 0.
        num_iterations = 0

        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        
        for batch_data, labels in data_iterator:
            # Move batch data and labels to the device
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = labels.to(device)
            
            num_iterations += 1
            total_iterations += 1
            optimizer.zero_grad()
            
            # Forward pass
            output = net(input_ids, attention_mask)
            loss = cross_el(output, labels)
            loss_sum += loss.item()
            
            # Calculate accuracy
            avg_loss = loss_sum / num_iterations
            acc = metrics_fn(output, labels)
            acc_sum += acc

            # Update progress bar with loss and learning rate
            current_lr = scheduler.get_last_lr()[0]  # Fetch the current learning rate
            data_iterator.set_postfix(loss=avg_loss, lr=current_lr)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return
        
        val_loss, val_acc = test(test_dl, net, cross_el, metrics_fn, device)
        print(f"Epoch {epoch + 1}: train loss: {avg_loss:.3f}, val loss: {val_loss:.3f}, "
              f"train acc: {acc_sum / num_iterations:.3f}, val acc: {val_acc:.3f}, lr: {current_lr:.6f}")

def test(test_dl, net, loss_fn, metrics_fn, device):
    net.eval()
    tot_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, labels in tqdm(test_dl, desc="Testing"):
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            tot_loss += loss.item()

            # Calculate predictions and accuracy
            predicted = torch.argmax(outputs, dim=-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    net.train() 
    return tot_loss / len(test_dl), accuracy
