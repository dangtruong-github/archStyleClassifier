import torch
from sklearn.metrics import confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"

def summary(loader, model, criterion):

    num_correct = 0
    num_samples = 0
    total_loss = 0
    
    model.eval()
    
    acc = 0
    
    cm = [[0, 0], [0, 0]] # tp 00, tn 01, fp 10, fn 11

    with torch.no_grad():
        for index, (data, label) in enumerate(loader):
            data = data.to(device=device)
            label = label.to(device=device)

            prob = model(data)

            pred = torch.argmax(prob, dim=1)

            if index % 10 == 9:
                pass

            num_correct += (pred == label).sum()
            num_samples += label.shape[0]
            loss = criterion(prob, label)

            total_loss += loss.item()
            
            try:
                conf_mat = confusion_matrix(label.cpu().numpy(), pred.cpu().numpy(), labels=[0,1]).ravel()
                
                #print(conf_mat)
                
                tn, fp, fn, tp = conf_mat

                cm[0][0] += tp
                cm[0][1] += tn
                cm[1][0] += fp
                cm[1][1] += fn
            
            except Exception as e:
                print(e)
            
        acc = float(num_correct)/float(num_samples) * 100.0
    return acc, total_loss, cm