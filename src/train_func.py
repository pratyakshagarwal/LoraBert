import torch
import torch.nn as nn
from tqdm import tqdm

def calculate_accuracy(logits, labels):
    """
    Calculate the accuracy of predictions.

    Parameters:
        logits (Tensor): The raw output predictions from the model (before applying softmax).
        labels (Tensor): The ground truth labels.

    Returns:
        float: The accuracy of the predictions.
    """
    # Get the predicted class labels by taking the argmax of logits
    predictions = torch.argmax(logits, dim=1)
    # Calculate the number of correct predictions
    correct = (predictions == labels).sum().item()
    # Calculate the accuracy as correct predictions divided by total samples
    accuracy = correct / labels.size(0)
    return accuracy

def train(train_loader, test_dl, net, epochs=5, total_iterations_limit=None, device='cpu', lr: float=1e-3):
    """
    Train the model for a specified number of epochs.

    Parameters:
        train_loader (DataLoader): The DataLoader for the training dataset.
        test_dl (DataLoader): The DataLoader for the test dataset.
        net (nn.Module): The neural network model to be trained.
        epochs (int): Number of training epochs. Defaults to 5.
        total_iterations_limit (int): The maximum number of iterations to run. If None, runs for the full epochs.
        device (str): The device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        lr (float): The learning rate for the optimizer. Defaults to 1e-3.

    Returns:
        None: The model is trained and saved, but nothing is returned.
    """
    # Loss function and optimizer
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    
    # Learning rate scheduler for dynamic learning rate adjustment
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr, total_steps=epochs * len(train_loader))
    
    # Function to calculate accuracy
    metrics_fn = calculate_accuracy
    total_iterations = 0

    # Move the model to the device (GPU/CPU)
    net.to(device)
    
    # Loop over the number of epochs
    for epoch in range(epochs):
        net.train()  # Set the model to training mode

        loss_sum, acc_sum = 0., 0.
        num_iterations = 0

        # Create a progress bar for the training loop
        data_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        # Set the iteration limit if specified
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        
        # Loop over the training data
        for batch_data, labels in data_iterator:
            # Move data and labels to the appropriate device
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = labels.to(device)

            num_iterations += 1
            total_iterations += 1
            optimizer.zero_grad()  # Clear the gradients

            # Forward pass
            output = net(input_ids, attention_mask)
            loss = cross_el(output, labels)
            loss_sum += loss.item()

            # Calculate the average loss and accuracy so far
            avg_loss = loss_sum / num_iterations
            acc = metrics_fn(output, labels)
            acc_sum += acc

            # Update the progress bar with loss and current learning rate
            current_lr = scheduler.get_last_lr()[0]  # Get the current learning rate
            data_iterator.set_postfix(loss=avg_loss, lr=current_lr)

            # Backward pass to compute gradients
            loss.backward()
            optimizer.step()  # Update model parameters
            scheduler.step()  # Adjust learning rate dynamically

            # Stop early if iteration limit is reached
            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return
        
        # Evaluate the model on the test data after each epoch
        val_loss, val_acc = test(test_dl, net, cross_el, metrics_fn, device)
        
        # Log the training and validation results for the current epoch
        print(f"Epoch {epoch + 1}: train loss: {avg_loss:.3f}, val loss: {val_loss:.3f}, "
              f"train acc: {acc_sum / num_iterations:.3f}, val acc: {val_acc:.3f}, lr: {current_lr:.6f}")

def test(test_dl, net, loss_fn, metrics_fn, device):
    """
    Evaluate the model on the test dataset.

    Parameters:
        test_dl (DataLoader): The DataLoader for the test dataset.
        net (nn.Module): The trained model to evaluate.
        loss_fn (function): The loss function used for evaluation.
        metrics_fn (function): The function used to calculate the evaluation metrics (e.g., accuracy).
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        tuple: The average loss and accuracy on the test dataset.
    """
    net.eval()  # Set the model to evaluation mode
    tot_loss = 0
    correct = 0
    total = 0

    # No gradients needed during evaluation
    with torch.no_grad():
        # Loop over the test data
        for batch_data, labels in tqdm(test_dl, desc="Testing"):
            # Move data and labels to the device
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = net(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)
            tot_loss += loss.item()

            # Calculate the accuracy
            predicted = torch.argmax(outputs, dim=-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Calculate average accuracy
    accuracy = correct / total
    net.train()  # Set the model back to training mode
    # Return the average loss and accuracy
    return tot_loss / len(test_dl), accuracy
