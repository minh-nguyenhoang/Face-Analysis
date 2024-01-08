import torch
import numpy as np
import logging
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, precision_score, average_precision_score, recall_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def loss_batch(model, loss_func, xb, age, gender, masked, emotion, race, skin, opt=None, metric=None, device='cpu'):
    # Generate predictions
    age, gender, masked, emotion, race, skin = age.to(device), gender.to(device), masked.to(device), emotion.to(device), race.to(device), skin.to(device)
    xb = xb.to(device)
    out = model(xb)
    # print(yb)
    # Calculate loss
    
    loss = loss_func(out, age, gender, masked, emotion, race, skin)
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
        # print(metric)
        metric_result = metric(out, age, gender, masked, emotion, race, skin)
    return loss.item(), len(xb), metric_result


def evaluate(model, loss_func, valid_dl, metric=None, device=None):
    with torch.no_grad():
        # Pass each batch through the model
        results = [loss_batch(model, loss_func, xb, age, gender, masked, emotion, race, skin, metric=metric, device=device)
                   for xb, age, gender, masked, emotion, race, skin in valid_dl]
        # Separate losses, counts and metrics
        losses, nums, metrics = zip(*results)
        # Total size of the data set
        total = np.sum(nums)
        # Avg, loss across batches
        avg_loss = np.sum(np.multiply(np.array(losses), np.array(nums))) / total
        if metric is not None:
            # Avg of metric across batches
            avg_metric = np.sum(np.multiply(torch.stack(metrics,dim=0).cpu().numpy(), np.array(nums))) / total
    return avg_loss, total, avg_metric


def accuracy(outputs, age, gender, masked, emotion, race, skin):
    out_age, out_gender, out_masked, out_emotion, out_race, out_skin = outputs

    age_pred = torch.sum(out_age > 0.5, dim=1)
    age = torch.sum(age, dim=1)
    gender_pred = torch.argmax(out_gender, dim=1) 
    masked_pred = torch.argmax(out_masked, dim=1)
    emotion_pred = torch.argmax(out_emotion, dim=1)
    race_pred = torch.argmax(out_race, dim=1)
    skin_pred = torch.argmax(out_skin, dim=1)
    
    age_acc = torch.mean( (age_pred == age).float())
    gender_acc = torch.mean((gender_pred == gender).float())
    masked_acc = torch.mean((masked_pred == masked).float())
    emotion_acc = torch.mean((emotion_pred == emotion).float())
    race_acc = torch.mean((race_pred == race).float())
    skin_acc = torch.mean((skin_pred == skin).float())

    return (age_acc + gender_acc + masked_acc + emotion_acc + race_acc + skin_acc) / 6

def trainer(epochs, model, loss_func, train_dl, valid_dl, opt_fn=None, lr=None, metric=accuracy, PATH='', device='cpu'):
    train_losses, val_losses, val_metrics = [], [], []
    max_val_acc = 0
    torch.cuda.empty_cache()
    # Instantiate the optimizer
    if opt_fn is None:
        opt_fn = torch.optim.Adam
    opt = opt_fn(model.parameters(),lr=lr,weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', patience= 8, min_lr =1e-4, verbose=True)

    for epoch in range(epochs):
        # Training
        model.train()
        for idx, (xb, age, gender, masked, emotion, race, skin) in enumerate(tqdm(train_dl)):
            train_loss = loss_batch(model, loss_func, xb, age, gender, masked, emotion, race, skin, opt, metric=metric, device=device)
            if idx == 50:
                model.eval()
                result = evaluate(model, loss_func=loss_func, valid_dl=valid_dl, metric=metric, device=device)
                model.train()
        # Evaluation
        model.eval()
        result = evaluate(model, loss_func=loss_func, valid_dl=valid_dl, metric=metric)
        val_loss, total, val_metric = result
        sched.step(val_loss)

        if max_val_acc < val_metric:
          torch.save(model.state_dict(), PATH + 'best_model.pth')

        # Record the loss and metric
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        
        # Print progress
        if metric is None:
            messages = 'Epoch [{} / {}], train_loss: {:4f}'\
                  .format(epoch + 1, epochs, train_loss)
        else:
            messages = 'Epoch [{} / {}], train_loss: {:4f}, val_loss:{:4f}, val_{}: {:4f}'\
                  .format(epoch + 1, epochs, train_loss, val_loss, metric.__name__, val_metric)
        # logger.info(messages)
    return train_losses, val_losses, val_metrics