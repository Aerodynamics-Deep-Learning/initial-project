import torch


def train(cfg_train:(dict), model:(torch.nn.Module), optimizer:(torch.optim), loss_fn, dataloaders:(list)):

    """
    This function is to train the model given an opitimizer, loss, and data.

    Args:
        cfg_train (dict):
            - 'cfg_loader' : The config dict for the loader that was previously used
            - 'device' : The device being used
            - 'dtype': Dtype to be used during the entire training process
        model (torch.nn.Model): The DL model being used
        optimizer (torch.optim): The optimizer being used to improve the model
        loss: The loss function
        dataloaders (list): List of dataloaders [dl_train, dl_val, dl_test] 
    
    Returns:
        collector_dict (dict): Dictionary of logs
        model (nn.Module): Trained model
    """

    # Unpack the cfg_train dict
    device = cfg_train['device']
    dtype = cfg_train.get('dtype', torch.float32)
    cfg_loader = cfg_train['cfg_loader']
    n_epoch = cfg_loader['n_epoch']
    n_train = cfg_loader['n_train']
    train_batch = cfg_loader['train_batch']


    dl_train, dl_val, dl_test = dataloaders[0], dataloaders[1], dataloaders[2] # Unpack the dataloaders
    iter_train = iter(dl_train); iter_val = iter(dl_val); iter_test = iter(dl_test) # Define the iterators for the dataloaders


    collector_dict = {
                    'actual': {
                        'val': {}
                    },
                    'preds': {
                        'val': {}
                    },
                    'losses': {
                        'train': [], 
                        'total_train': [],
                        'val': [], 
                        'test': []
                    }
                    } # Initialize the collector dict to collect the predictions and losses
    
    # Before training, infer the amount of training batches per epoch
    n_train_batch_per_epoch = int(n_train/train_batch)


    for epoch in range(1, n_epoch +1):

        # Training loop for the epoch

        model.train() # Set the model to training mode
        total_train_loss = 0.0 # Set the training loss to 0

        for _ in range(n_train_batch_per_epoch):

            # Get the batch
            try:
                batch= next(iter_train)
            except StopIteration:
                # If the dataset is exhausted, reset the iterator for the next epoch
                raise StopIteration("The training dataset has been exhausted. There must be a fault.")
            
            # Get the values from the batch, move them to the necessary device with necessary dtype
            shape_input, param_input, target, airfoil_name = batch['geometry'].to(device= device, dtype= dtype), batch['cond'].to(device= device, dtype= dtype), batch['perf_coeffs'].to(device= device, dtype= dtype), batch['name'] 

            # Zero them grads
            optimizer.zero_grad()

            # Get the predictions
            pred = model(shape_input= shape_input, param_input= param_input)

            # Get the loss 
            loss = loss_fn(target, pred)
            
            # Backprop
            loss.backward()
            optimizer.step()

            # Log results
            print(f'Epoch {epoch} | Train run {_+1} loss: {loss}')
            total_train_loss += loss.item()
            collector_dict['losses']['train'].append(loss.item())

        print(f'Epoch {epoch} | Total train loss: {total_train_loss}')
        collector_dict['losses']['total_train'].append(total_train_loss)
        
        # Validation step

        # Set model to eval
        model.eval()

        ctual_vals = []
        pred_vals = []

        with torch.no_grad():

            try:
                batch= next(iter_val)
            except StopIteration:
                # If the dataset is exhausted, reset the iterator for the next epoch
                raise StopIteration("The validation dataset has been exhausted. There must be a fault.")
        
            shape_input, param_input, target, airfoil_name = batch['geometry'].to(device= device, dtype= dtype), batch['cond'].to(device= device, dtype= dtype), batch['perf_coeffs'].to(device= device, dtype= dtype), batch['name'] 

            pred = model(shape_input= shape_input, param_input= param_input)

            loss = loss_fn(target, pred)

            # Log the validation loss
            print(f'Epoch {epoch} | Val loss: {loss}')
            collector_dict['losses']['val'].append(loss.item())

            # Log the validation predictions and actual vals
            collector_dict['preds']['val'][f'{epoch}'] = pred
            collector_dict['actual']['val'][f'{epoch}'] = target
    
    
    # Testing step

    with torch.no_grad():

        try:
            batch= next(iter_test)
        except StopIteration:
            # If the dataset is exhausted, reset the iterator for the next epoch
            raise StopIteration("The validation dataset has been exhausted. There must be a fault.")
        
        shape_input, param_input, target, airfoil_name = batch['geometry'].to(device= device, dtype= dtype), batch['cond'].to(device= device, dtype= dtype), batch['perf_coeffs'].to(device= device, dtype= dtype), batch['name'] 

        pred = model(shape_input= shape_input, param_input= param_input)

        loss = loss_fn(target, pred)

        # Log the test loss
        print(f'Test loss: {loss}')
        collector_dict['losses']['test'].append(loss.item())

        # Log the validation predictions and actual vals
        collector_dict['preds']['test'] = pred
        collector_dict['actual']['test'] = target
    
    """
    Train, validation, and test has been finished. Now, final changes will be made to the necessary parts and outs will be given.
    """
    
    # Convert the lists to tensors
    collector_dict['losses']['train'] = torch.tensor(collector_dict['losses']['train'], dtype=torch.float32)
    collector_dict['losses']['total_train'] = torch.tensor(collector_dict['losses']['total_train'], dtype=torch.float32)
    collector_dict['losses']['val'] = torch.tensor(collector_dict['losses']['val'], dtype=torch.float32)
    collector_dict['losses']['test'] = torch.tensor(collector_dict['losses']['test'], dtype=torch.float32)

    return collector_dict, model



def train_one_epoch(model, optimizer, criterion, inputs, targets, batch_size= 32, device= None):

    """
    Deprecated.
    """


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

        loss = criterion(outputs, batch_targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_inputs.size(0)

        print(f"Rows {start_index} to {start_index+batch_size-1}, loss: " + str(loss.item()))

    return running_loss / n_samples