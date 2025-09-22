import torch
import yaml

import os
import sys
current_directory = os.getcwd()
root_directory = os.path.abspath(os.path.join(current_directory, "..", "..", ".."))
sys.path.append(root_directory)
from data.physical_simulation_Dataset import CylinderDynamicsDataset
from model.physical_simulation.Cylinder.cylinder_model import CYLINDER_ROM

trainer_directory = os.path.abspath(os.path.join(current_directory, ".."))
sys.path.append(trainer_directory)
from trainer import set_seed, train_jointly_forward_model, save_training_log


def main():
    set_seed(42)
    torch.set_default_dtype(torch.float32)
    
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using {device} device")
    if torch.cuda.is_available():
        print(f"[INFO] {torch.cuda.get_device_properties(0)}")
    
    # Load configuration
    config_path = "cylinder.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print("[INFO] Starting Cylinder Flow Model Training")
    print(f"[INFO] Configuration: {config}")
    
    print("\n" + "="*50)
    print("TRAINING FORWARD MODEL")
    print("="*50)
    
    # Load dynamics dataset
    cyl_train_dataset = CylinderDynamicsDataset(
                data_path="../../../data/physical_simulation/cylinder/cylinder_train_data.npy",
                seq_length = config['seq_length'],
                mean=None,
                std=None)
    
    cyl_val_dataset = CylinderDynamicsDataset(
                data_path="../../../data/physical_simulation/cylinder/cylinder_val_data.npy",
                seq_length = config['seq_length'],
                mean=cyl_train_dataset.mean,
                std=cyl_train_dataset.std)
    
    # Create forward model
    forward_model = CYLINDER_ROM()
    
    print("\n" + "="*50)
    print("JOINT TRAINING")
    print("="*50)
    
    train_loss, val_loss = train_jointly_forward_model(
        forward_model=forward_model,
        train_dataset=cyl_train_dataset,
        val_dataset=cyl_val_dataset,
        model_save_folder=config['save_folder'],
        learning_rate=config['learning_rate'],
        lamb=config['lamb'],
        lamb_mi=config['alpha'],
        lamb_entropy=config['beta'],
        batch_size=config['batch_size'],
        num_epochs=config['num_epochs'],
        decay_step=config['decay_step'],
        decay_rate=config['decay_rate'],
        device=device,
        patience=config['patience']
    )
    
    save_training_log(train_loss, val_loss, f"{config['save_folder']}/losses", 0)

if __name__ == "__main__":
    main()