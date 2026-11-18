import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from model import HeartbeatCNN
train=pd.read_csv('../data/mitbih_train.csv',header=None).values
test=pd.read_csv('../data/mitbih_test.csv',header=None).values
X_train,y_train=train[:,:187],train[:,187].astype(int)
X_test,y_test=test[:,:187],test[:,187].astype(int)
X_train=(X_train-X_train.mean(1,keepdims=True))/(X_train.std(1,keepdims=True)+1e-8)
X_test=(X_test-X_test.mean(1,keepdims=True))/(X_test.std(1,keepdims=True)+1e-8)
X_train_t=torch.tensor(X_train,dtype=torch.float32).unsqueeze(1)
X_test_t=torch.tensor(X_test,dtype=torch.float32).unsqueeze(1)
y_train_t=torch.tensor(y_train,dtype=torch.long); y_test_t=torch.tensor(y_test,dtype=torch.long)
model=HeartbeatCNN()
weights=compute_class_weight('balanced',classes=np.unique(y_train),y=y_train)
criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32))
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
for epoch in range(15):
    model.train(); perm=torch.randperm(len(X_train_t))
    for i in range(0,len(X_train_t),256):
        idx=perm[i:i+256]; loss=criterion(model(X_train_t[idx]),y_train_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): acc=(model(X_test_t).argmax(1)==y_test_t).float().mean().item()
    print(f"Epoch {epoch+1} Test Acc {acc*100:.2f}%")
torch.save(model.state_dict(),'../checkpoints/heartbeat_cnn.pt')
