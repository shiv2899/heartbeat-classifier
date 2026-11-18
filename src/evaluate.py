import pandas as pd, torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns
from model import HeartbeatCNN
test=pd.read_csv('../data/mitbih_test.csv',header=None).values
X_test,y_test=test[:,:187],test[:,187].astype(int)
X_test=(X_test-X_test.mean(1,keepdims=True))/(X_test.std(1,keepdims=True)+1e-8)
Xt=torch.tensor(X_test,dtype=torch.float32).unsqueeze(1)
model=HeartbeatCNN(); model.load_state_dict(torch.load('../checkpoints/heartbeat_cnn.pt')); model.eval()
with torch.no_grad(): preds=model(Xt).argmax(1).numpy()
print(classification_report(y_test,preds,target_names=['N','S','V','F','Q']))
cm=confusion_matrix(y_test,preds)
plt.figure(figsize=(7,6)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['N','S','V','F','Q'],yticklabels=['N','S','V','F','Q'])
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Heartbeat Confusion Matrix')
plt.savefig('../results/confusion_matrix.png',dpi=150); plt.show()
