from os.path import isfile
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np


def benchmark(PS_of_params, test_PS, test_params):
    # Main benchmark z = 8 at k = 0.192 h/Mpc
    # Other benchmark: "all 78 numbers from old emulator",
    # or all HERA-k at all HERA-z, or something likelihood
    # related such as L_true vs L_emu.
    # Also consider TS emulator etc.
    raise NotImplementedError

# The main power spectrum emulator, also for RSD ratios
class poweremu():
    def __init__(self, loadfile=None, hidden_layer_sizes=None, preprocesss_log_x=False,
                 preprocess_y=True, offset=1, max_iter=10, **kwargs):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = (100,100,100,100)
        self.mlp = make_pipeline(StandardScaler(), MLPRegressor(
            # Changeable non-defaults
            hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter,
            # Mandatory non-defaults
            verbose=True, validation_fraction=0, warm_start=True,
            # Defaults
            #activation='relu', early_stopping=False,
            #alpha=1e-4, solver='adam', learning_rate='constant',
            **kwargs))
        if preprocesss_log_x:
            self.preprocess_x = lambda x: np.log(x)
            self.inv_preprocess_x = lambda x: np.exp(x)
        else:
            self.preprocess_x = lambda x: x
            self.inv_preprocess_x = lambda x: x
        if preprocess_y:
            self.preprocess_y = lambda y: np.log(y+offset)
            self.inv_preprocess_y = lambda y: np.exp(y)-offset
        else:
            self.preprocess_y = lambda y: y
            self.inv_preprocess_y = lambda y: y
        if loadfile is None:
            print("Not loading from file.")
        elif isfile(loadfile):
            self.load(loadfile)
        else:
            assert False, ("loadfile", loadfile, "not found.")
    def load(self, loadfile):
        self.mlp = joblib.load(loadfile)
        #print("Loaded from", loadfile)
    def save(self, loadfile):
        joblib.dump(self.mlp, loadfile)
        print("Saved to", loadfile)
    def train(self, input_0, output_0):
        # Step 0: Take the ln of x
        input_1 = self.preprocess_x(input_0)
        # Step 1: Take the ln of y+1
        output_1 = self.preprocess_y(output_0)
        # Step 2: Train emulator incl. scaler
        return self.mlp.fit(input_1, output_1)
    def predict(self, x):
        single_point = True if len(np.shape(x))==1 else False
        x = [x] if single_point else x
        x1 = self.preprocess_x(x)
        y1 = self.mlp.predict(x1)
        y0 = self.inv_preprocess_y(y1)
        y0 = y0[0] if single_point else y0
        return y0

#build MLP regressor in pytorch
import torch 
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_hidden = 1, out_dim = 1):
        super(MLP, self).__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_hidden = [nn.Linear(hidden_dim, hidden_dim) for i in range(n_hidden)]
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        for fc in self.fc_hidden:
            x = self.relu(fc(x))
        x = self.fc_out(x)

        return x

class poweremu_torch(nn.Module):
    def __init__(self, network, network_opt, train_opt, learning_rate=1e-3, device="cpu"):
        super(poweremu_torch, self).__init__()
        self.network = network #MLP
        self.network_opt = network_opt #dictionary of MLP args
        self.model = network(**network_opt)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.train_opt = train_opt #dictionary of training args (learning rate, epochs, etc.)
        self.loss = []

        self.multi_gpu = torch.cuda.device_count() > 1 
        self.device = device #"cpu" or "cuda:i"
        
    def train(self, train_data):
        
        self.model.train()
        for epoch in range(self.train_opt["epochs"]):
            loss_epoch = torch.tensor(0., device=self.device)
            for i,(input,target) in enumerate(train_data):
                
                if self.train_opt.get("log_indices") is not None: 
                    input[:, self.train_opt["log_indices"]] = torch.log10(input[:, self.train_opt["log_indices"]])
                if (self.train_opt.get("log_output") is not None) and (self.train_opt.get("log_output")):
                    target = torch.log10(target)

                predicted = self.model(input)
                #loss_batch = torch.nn.MSELoss()(predicted, output)
                
                loss_batch = torch.nanmean(torch.square(predicted - target))
                

                self.optimizer.zero_grad()
                loss_batch.backward()
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                self.optimizer.step()

                if self.multi_gpu:
                    torch.distributed.all_reduce(tensor=loss_batch, op=torch.distributed.ReduceOp.AVG)
                
                loss_epoch += loss_batch / len(train_data)


            self.loss.append(loss_epoch.item())

            print(f"[{self.device}] Epoch {epoch} | Loss: {loss_epoch.item()}", flush=True)

            if loss_epoch == torch.min(torch.tensor(self.loss)):
                print("Saving model! (train print)", flush=True)
                #self.save_network("best_model.pth")
            else:
                print("Not saving model! (train print)", flush=True)
            
            if self.multi_gpu:
                torch.distributed.barrier()
    
    @torch.no_grad()
    def predict(self, x):
        self.model.eval()
        if self.train_opt.get("log_indices") is not None:
            x[:, self.train_opt["log_indices"]] = torch.log10(x[:, self.train_opt["log_indices"]])

        y = self.model(x)
        
        if (self.train_opt.get("log_output") is not None) and (self.train_opt.get("log_output")):
            y = 10**y
        return y

    def save_network(self, path):
        if not self.multi_gpu:
            torch.save(
                obj = dict(
                    network_opt = self.network_opt,
                    model = self.model.state_dict(), 
                    optimizer = self.optimizer.state_dict(),
                    train_opt = self.train_opt,
                    #ema = self.ema.state_dict(),
                    loss = self.loss,
                    ),
                    f = path
                    )
        else:
            if str(self.device) == "cuda:0":
                print("Saving model!", flush=True)
                torch.save(
                    obj = dict(
                        network_opt = self.network_opt,
                        model = self.model.module.state_dict(), 
                        optimizer = self.optimizer.state_dict(),
                        train_opt = self.train_opt,
                        #ema = self.ema.state_dict(),
                        loss = self.loss,
                        ),
                        f = path
                        )

    def load_network(self, path):
        loaded_state = torch.load(path, map_location=self.device)
        self.network_opt = loaded_state['network_opt']
        self.model = self.network(**self.network_opt)
        self.model.load_state_dict(loaded_state['model'])
        if self.multi_gpu:
            self.model.to(self.device)
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.rank])
        self.train_opt = loaded_state['train_opt']
        self.optimizer.load_state_dict(loaded_state['optimizer'])
        self.loss = loaded_state['loss']
             
    
if __name__ == "__main__":
    # Example usage
    # Load data
    
    from scipy.io import loadmat
    import numpy as np
    from tools import gen_training

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    z_array = loadmat("/home/dp331/dp331/dc-poch1/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_z_mat.mat")["z21cm"][0]
    z_mask = (z_array >= 7) & (z_array <= 27)
    z_array = z_array[z_mask].astype(np.float32)
    #z_array = torch.from_numpy(z_array).to(device)

    k_array = loadmat("/home/dp331/dp331/dc-poch1/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_k_mat.mat")["ks"][0]
    k_mask = k_array >= 8.5e-2
    k_array = k_array[k_mask].astype(np.float32)
    #k_array = torch.from_numpy(k_array).to(device)

    parameters = loadmat("/home/dp331/dp331/dc-poch1/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_parameters_mat.mat")["parameters"]
    parameters = np.delete(parameters, [6,-2,-1], axis=1).astype(np.float32)
    #parameters = torch.from_numpy(parameters).to(device)
    
    Deltak = loadmat("/home/dp331/dp331/dc-poch1/CosmicDawnSynergies/data/models_21cmSim/HERA_IDR4_Emulator_Data/hera_Deltak_mat.mat")["combined_Deltaks"][:,z_mask,:][:,:,k_mask]
    Deltak = Deltak.astype(np.float32)
    #Deltak = torch.from_numpy(Deltak).to(device)
    
    h = 0.6704
    input_train, input_test, output_train, output_test = train_test_split(parameters, Deltak, test_size=0.2, random_state=42)
        
    input_train_interp, output_train_interp = gen_training(n_over=1, params=input_train, data=output_train, zlow=z_array.min(), zhigh=z_array.max(), klow=k_array.min()/h, khigh=k_array.max()/h)
    #find indices where output_train_interp is zero and delete them (also be careful of nans and infs)
    zero_indices = np.argwhere(output_train_interp == 0)
    output_train_interp = np.delete(output_train_interp, zero_indices[:,0], axis=0)
    input_train_interp = np.delete(input_train_interp, zero_indices[:,0], axis=0)
    
    
    input_train_interp = torch.from_numpy(input_train_interp.astype(np.float32)).to(device)
    output_train_interp = torch.from_numpy(output_train_interp.astype(np.float32)).to(device)




    # Train
    from torch.utils.data import DataLoader, TensorDataset
    train_data = TensorDataset(input_train_interp, output_train_interp)
    train_loader = DataLoader(train_data, batch_size=1000, shuffle=True)


    network_opt = {
        "in_dim": input_train_interp.shape[1],
        "hidden_dim": 100,
        "n_hidden": 3,
        "out_dim": 1
    }
    train_opt = {
        "epochs": 40,
        "log_indices": 2+np.array([0,1,2,3,7]), #+2 because we add z and k
        "log_output": True
    }

    Emulator = poweremu_torch(network=MLP, network_opt=network_opt, train_opt=train_opt, learning_rate=1e-5, device=device)
    Emulator.train(train_loader)
    
    
    
    
    