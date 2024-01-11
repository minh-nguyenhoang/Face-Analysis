import torch
import numpy as np
import logging
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, precision_score, average_precision_score, recall_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def loss_batch(model, loss_func, xb, age, gender, masked, emotion, race, skin, opt=None, metric=None, device='cpu', eval=False):
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
        metric_result, age_acc, gender_acc, masked_acc, emotion_acc, race_acc, skin_acc = metric(out, age, gender, masked, emotion, race, skin, eval)
    return loss.item(), len(xb), metric_result, age_acc, gender_acc, masked_acc, emotion_acc, race_acc, skin_acc


def evaluate(model, loss_func, valid_dl, metric=None, device=None, eval=False):

    with torch.no_grad():
        # Pass each batch through the model
        losses, nums, metrics, age_metrics, gender_metrics, masked_metrics, emotion_metrics, race_metrics, skin_metrics = \
        [], [], [], [], [], [], [], [], []

        preds = {
            'age': [],
            'gender': [],
            'masked': [],
            'emotion': [],
            'race': [],
            'skin': []
        }
        labels = {
            'age': [],
            'gender': [],
            'masked': [],
            'emotion': [],
            'race': [],
            'skin': []
        }
        for idx, (xb, age, gender, masked, emotion, race, skin) in enumerate(tqdm(valid_dl)):
            age, gender, masked, emotion, race, skin = age.to(device), gender.to(device), masked.to(device), emotion.to(device), race.to(device), skin.to(device)
            xb = xb.to(device)
            out = model(xb)
            out_age, out_race, out_gender, out_masked, out_emotion, out_skin = out

            age_pred = torch.argmax(out_age, dim=1)
            gender_pred = torch.sigmoid(out_gender)  > 0.5
            masked_pred = torch.sigmoid(out_masked)  > 0.5

            emotion_pred = torch.argmax(out_emotion, dim=1)
            race_pred = torch.argmax(out_race, dim=1)
            skin_pred = torch.argmax(out_skin, dim=1)
            preds['age'].append(age_pred)
            preds['gender'].append(gender_pred)
            preds['masked'].append(masked_pred)
            preds['race'].append(race_pred)
            preds['emotion'].append(emotion_pred)
            preds['skin'].append(skin_pred)

            labels['age'].append(age)
            labels['gender'].append(gender)
            labels['masked'].append(masked)
            labels['race'].append(race)
            labels['emotion'].append(emotion)
            labels['skin'].append(skin)
            # print(yb)
            # Calculate loss
            loss_batch = loss_func(out, age, gender, masked, emotion, race, skin)
            if metric is not None:
                # Compute the metric
                # print(metric)
                metric_result, age_acc, gender_acc, masked_acc, emotion_acc, race_acc, skin_acc = metric(out, age, gender, masked, emotion, race, skin, eval)
            losses.append(loss_batch.cpu())
            nums.append(len(xb))
            metrics.append(metric_result.cpu())
            age_metrics.append(age_acc.cpu())
            gender_metrics.append(gender_acc.cpu())
            masked_metrics.append(masked_acc.cpu())
            emotion_metrics.append(emotion_acc.cpu())
            race_metrics.append(race_acc.cpu())
            skin_metrics.append(skin_acc.cpu())

        for k, v in preds.items():
            preds[k] = torch.cat(v, dim=0).cpu().tolist()
        for k, v in labels.items():
            labels[k] = torch.cat(v, dim=0).cpu().tolist()

        # Separate losses, counts and metrics
        # losses, nums, metrics, age_metrics, gender_metrics, masked_metrics, emotion_metrics, race_metrics, skin_metrics = zip(*results)
        # Total size of the data set
        total = np.sum(nums)
        # Avg, loss across batches
        avg_loss = np.sum(np.multiply(np.array(losses), np.array(nums))) / total

        for k in preds.keys():
            print(f'{k} - report\n')
            print(classification_report(labels[k], preds[k]))
            print('\n\n')
        
        if metric is not None:
            # Avg of metric across batches
            avg_metric = np.sum(np.multiply(torch.stack(metrics,dim=0).cpu().numpy(), np.array(nums))) / total
            avg_age_metric = np.sum(np.multiply(torch.stack(age_metrics,dim=0).cpu().numpy(), np.array(nums))) / total
            avg_gender_metric = np.sum(np.multiply(torch.stack(gender_metrics,dim=0).cpu().numpy(), np.array(nums))) / total
            avg_masked_metric = np.sum(np.multiply(torch.stack(masked_metrics,dim=0).cpu().numpy(), np.array(nums))) / total
            avg_emotion_metric = np.sum(np.multiply(torch.stack(emotion_metrics,dim=0).cpu().numpy(), np.array(nums))) / total
            avg_race_metric = np.sum(np.multiply(torch.stack(race_metrics,dim=0).cpu().numpy(), np.array(nums))) / total
            avg_skin_metric = np.sum(np.multiply(torch.stack(skin_metrics,dim=0).cpu().numpy(), np.array(nums))) / total
             
    return avg_loss, total, avg_metric, avg_age_metric, avg_gender_metric, avg_masked_metric, avg_emotion_metric, avg_race_metric, avg_skin_metric


def accuracy(outputs, age, gender, masked, emotion, race, skin, eval):
    out_age, out_race, out_gender, out_masked, out_emotion, out_skin = outputs
    # age_pred = torch.sum(out_age > 0.5, dim=1)
    age_pred = torch.argmax(out_age, dim=1)
    # age = torch.sum(age, dim=1)
    gender_pred = torch.sigmoid(out_gender)  > 0.5
    masked_pred = torch.sigmoid(out_masked)  > 0.5

    emotion_pred = torch.argmax(out_emotion, dim=1)
    race_pred = torch.argmax(out_race, dim=1)
    skin_pred = torch.argmax(out_skin, dim=1)
    
    age_acc = torch.mean( (age_pred == age).float())
    gender_acc = torch.mean((gender_pred == gender).float())
    masked_acc = torch.mean((masked_pred == masked).float())
    emotion_acc = torch.mean((emotion_pred == emotion).float())
    race_acc = torch.mean((race_pred == race).float())
    skin_acc = torch.mean((skin_pred == skin).float())

    return (age_acc + gender_acc + masked_acc + emotion_acc + race_acc + skin_acc) / 6, age_acc, gender_acc, masked_acc, emotion_acc, race_acc, skin_acc

def trainer(epochs, model, loss_func, train_dl, valid_dl, opt_fn=None, lr=None, metric=accuracy, PATH='', device='cpu'):
    train_losses, train_metrics = [], []
    val_losses, val_metrics = [], []
    max_val_acc = 0
    min_val_loss = 30
    torch.cuda.empty_cache()
    # Instantiate the optimizer
    if opt_fn is None:
        opt_fn = torch.optim.Adam
    opt = opt_fn(model.parameters(),lr=lr,weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=opt, mode='min', patience= 8, min_lr =1e-4, verbose=True)

    for epoch in range(epochs):
        avg_loss = []
        avg_train_metric = []
        m_age_acc, m_gender_acc, m_masked_acc, m_emotion_acc, m_race_acc, m_skin_acc = [], [], [], [], [], []
        # Training
        model.train()
        for idx, (xb, age, gender, masked, emotion, race, skin) in enumerate(tqdm(train_dl)):
            train_loss, nums, train_metric, _,_,_,_,_,_ = loss_batch(model, loss_func, xb, age, gender, masked, emotion, race, skin, opt, metric=metric, device=device)
            avg_loss.append(train_loss)
            avg_train_metric.append(train_metric)

        # Evaluation
        mean_loss = torch.mean(torch.Tensor(avg_loss))
        mean_metric = torch.mean(torch.Tensor(avg_train_metric))
        
        
        model.eval()
        result = evaluate(model, loss_func=loss_func, valid_dl=valid_dl, metric=metric, device=device, eval=True)
        val_loss, total, val_metric, age_metric, gender_metric, masked_metric, emotion_metric, race_metric, skin_metric = result
        sched.step(val_loss)

        if max_val_acc < val_metric:
            max_val_acc = val_metric
            torch.save(model.state_dict(), PATH + 'best_model_acc.pth')
        if min_val_loss >= val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + 'best_model_loss.pth')

        # Record the loss and metric
        train_losses.append(mean_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        # print(train_loss)
        # Print progress
        if metric is None:
            messages = 'Epoch [{} / {}], train_loss: {:4f}'\
                .format(epoch + 1, epochs, train_loss)
        else:
            messages = 'Epoch [{} / {}], train_loss: {:4f}, train_metric: {:4f}, val_loss:{:4f}, val_{}: {:4f}, age_acc: {:4f}, gender_acc: {:4f}, masked_acc: {:4f}, emotion_acc: {:4f}, race_acc: {:4f}, skin_acc: {:4f}'\
                .format(epoch + 1, epochs, mean_loss, mean_metric, val_loss, metric.__name__, val_metric, age_metric, gender_metric, masked_metric, emotion_metric, race_metric, skin_metric)
            # logger.info(messages)
        print(messages)
    return train_losses, val_losses, val_metrics
