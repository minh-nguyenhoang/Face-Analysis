import torch
import numpy as np
import logging
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def loss_batch(model, loss_func, xb, yb, opt=None, metric=None, device='cuda'):
    # Generate predictions
    yb = yb.to(device)
    xb = xb.to(device)
    out = model(xb)
    # print(yb)
    # Calculate loss
    
    loss = loss_func(xb,yb)
    # print(loss)
    if opt is not None:
        # Compute gradients
        loss.backward()
        # Update parameters
        opt.step()
        # Reset gradients
        opt.zero_grad()
    metric_result = None
    if metric is not None:
        # Compute the metric
        metric_result = metric(out, yb)
    return loss.item(), len(xb), metric_result


def evaluate(model, loss_func, valid_dl, metric=None):
    with torch.no_grad():
        # Pass each batch through the model
        results = [loss_batch(model, loss_func, xb, yb, metric=metric)
                   for xb, yb in valid_dl]
        # Separate losses, counts and metrics
        losses, ob_loss, dist_loss, nums, metrics = zip(*results)
        # Total size of the data set
        total = np.sum(nums)
        # Avg, loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        if metric is not None:
            # Avg of metric across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def trainer(epochs, model, loss_func, train_dl, valid_dl, opt_fn=None, lr=None, metric=None, PATH=''):
    train_losses, val_losses, val_metrics = [], [], []
    max_val_acc = 0
    torch.cuda.empty_cache()
    # Instantiate the optimizer
    if opt_fn is None:
        opt_fn = torch.optim.SGD
    opt = opt_fn(model.parameters(),lr=lr,weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', patience= 8, min_lr =1e-4, verbose=True)

    for epoch in range(epochs):
        # Training
        model.train()
        for xb, yb in tqdm(train_dl):
            train_loss, pb_loss, dist_loss,_,_ = loss_batch(model, loss_func, xb, yb, opt)
            # print(train_loss)
        # Evaluation
        model.eval()
        result = evaluate(model, loss_func=loss_func, valid_dl=valid_dl, metric=metric)
        val_loss, total, val_metric = result
        sched.step(val_loss)

        if max_val_acc < val_metric:
          torch.save(model.state_dict(), PATH + 'best_corrector.pth')

        # Record the loss and metric
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        
        # Print progress
        if metric is None:
            messages = 'Epoch [{} / {}], train_loss: {:4f}, pb_loss: {:4f}, dist_loss: {:4f}'\
                  .format(epoch + 1, epochs, train_loss, pb_loss, dist_loss)
        else:
            messages = 'Epoch [{} / {}], train_loss: {:4f}, val_loss:{:4f}, val_{}: {:4f}'\
                  .format(epoch + 1, epochs, train_loss, val_loss, metric.__name__, val_metric)
        logger.info(messages)
    return train_losses, val_losses, val_metrics