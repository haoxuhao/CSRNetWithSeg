import h5py
import torch
import shutil
import os
import subprocess

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best, save_path, tag=None, best_mae = None):
    if tag != None:
        filename = "%s_backup.pth.tar"%tag
        best_filename = "%s_best.pth.tar"%tag
    else:
        filename = "backup.pth.tar"
        best_filename = "best.pth.tar"

    if best_mae is not None:
        best_filename = "mae_%.2f_best.pth.tar"%best_mae

    torch.save(state, os.path.join(save_path, filename))

    if is_best:
        rets = subprocess.getstatusoutput("rm -f *best.pth.tar")
        print(rets)
        shutil.copyfile(os.path.join(save_path, filename), os.path.join(save_path,best_filename))
