import torch

def compute_loss_with_regularization(model, criterion, outputs, targets, lambda_reg=1e-4):
    base_loss = criterion(outputs, targets)
    
    # Manual L2 regularization on all parameters
    l2_reg = torch.tensor(0.0, device=outputs.device)
    for param in model.parameters():
        if param.requires_grad:
            l2_reg += torch.norm(param, p=2) ** 2
    
    return base_loss + lambda_reg * l2_reg

def train_one_epoch(model, optimizer, criterion, inputs, targets, batch_size= 32, device= None, lambda_reg = 0.0):

    model.train()

    running_loss = 0.0

    assert inputs.size(0) == targets.size(0), "Input and target datasets must have same amount of samples"

    n_samples = inputs.size(0)

    for start_index in range(0, n_samples, batch_size):

        end_index = start_index + batch_size

        batch_inputs = inputs[start_index:end_index].to(device)
        batch_targets = targets[start_index:end_index].to(device)

        optimizer.zero_grad()
        outputs = model(batch_inputs)

        if lambda_reg > 0:
            loss = compute_loss_with_regularization(model, criterion, outputs, batch_targets, lambda_reg)

        else:
            loss = criterion(outputs, batch_targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_inputs.size(0)

        print(loss.item())
        print(f"Rows: {start_index} to {start_index+batch_size-1} calculating")

    return running_loss / n_samples

