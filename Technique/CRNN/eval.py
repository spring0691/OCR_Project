from tqdm import tqdm
import engine
import config


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)
        batch_preds, loss = model(**data)
        fin_loss += loss.item()
        fin_preds.append(batch_preds)
    return fin_preds, fin_loss / len(data_loader)

model = 
test_loader = 
valid_preds, test_loss = eval_fn(model, test_loader)