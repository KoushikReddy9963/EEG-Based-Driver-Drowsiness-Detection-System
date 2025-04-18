import torch
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.optim as optim
from InterpretableCNN import InterpretableCNN

torch.cuda.empty_cache()
torch.manual_seed(0)

def run():
    filename = r'dataset.mat'
    tmp = sio.loadmat(filename)
    xdata = np.array(tmp['EEGsample'])
    label = np.array(tmp['substate'])   
    subIdx = np.array(tmp['subindex'])  

    label = label.astype(int)
    subIdx = subIdx.astype(int)
    samplenum = label.shape[0]

    channelnum = 30
    subjnum = 11           
    samplelength = 3      
    sf = 128               
    lr = 1e-3            
    batch_size = 50      
    n_epoch = 20

    ydata = np.zeros(samplenum, dtype=np.longlong)
    for i in range(samplenum):
        ydata[i] = label[i]
    print("Unique labels:", np.unique(ydata))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = np.zeros(subjnum) 

    for i in range(1, subjnum + 1):
        trainindx = np.where(subIdx != i)[0]
        xtrain = xdata[trainindx]
        x_train = xtrain.reshape(xtrain.shape[0], 1, channelnum, samplelength * sf)
        y_train = ydata[trainindx]

        testindx = np.where(subIdx == i)[0]
        xtest = xdata[testindx]
        x_test = xtest.reshape(xtest.shape[0], 1, channelnum, samplelength * sf)
        y_test = ydata[testindx]

        # Print data shapes for verification
        # print(f"Subject {i}: x_train shape: {x_train.shape}, x_test shape: {x_test.shape}")
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train)
        )
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        my_net = InterpretableCNN().float().to(device) 
        optimizer = optim.Adam(my_net.parameters(), lr=lr)
        loss_class = torch.nn.NLLLoss().to(device)

        for epoch in range(n_epoch):
            my_net.train()  
            for data in train_loader:
                inputs, labels = data
                input_data = inputs.to(device)
                class_label = labels.to(device)

                optimizer.zero_grad()
                class_output = my_net(input_data)
                err = loss_class(class_output, class_label)
                err.backward() 
                optimizer.step() 

            # Print training progress every 5 epochs
            # if epoch % 5 == 0:
            #     print(f"Subject {i}, Epoch {epoch}, Loss: {err.item():.4f}")

        
        my_net.eval() 
        with torch.no_grad():
            x_test = torch.FloatTensor(x_test).to(device)
            answer = my_net(x_test)
            probs = answer.cpu().numpy()
            preds = probs.argmax(axis=-1) 
            acc = accuracy_score(y_test, preds)

            # Determine status with a threshold of 0.5
            mean_pred = np.mean(preds)
            status = "Active" if mean_pred < 0.49 else "Drowsy"
            print(f"Subject {i} Accuracy: {acc:.4f} | Status: {status}")
            results[i - 1] = acc

    # --- Final Results ---
    print('Mean accuracy:', np.mean(results))

if __name__ == '__main__':
    run()