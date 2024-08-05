import torch
from base.dataset import ADT_Dataset
from model.simple import Simple_Eye_Gaze_MLP, Simple_Eye_Gaze_Loss, Simple_Trajectory_MLP, Simple_Trajectory_Loss
from base.utils import load_config
from base.metrics import Average_Gaze_Angular_Error, Average_Traj_Error
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train():
    config = load_config("./configs/config.yaml")
    batch_size = config['batch_size']
    epochs = config['epochs']
    use_gpu = config['use_gpu']
    model_name = config['model']
    num_workers = config['num_workers']
    lr = config['learning_rate']
    weight_decay = config['weight_decay']

    len_per_input_seq = config['len_per_input_seq']
    len_per_output_seq = config['len_per_output_seq']
    interval = config['interval']
    frame_stride = config['frame_stride']

    hidden_dim = config['hidden_dim']

    if use_gpu:
        assert torch.cuda.is_available()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    adt_dataset = ADT_Dataset(
        len_per_input_seq=len_per_input_seq, 
        len_per_output_seq=len_per_output_seq, 
        interval=interval, 
        stride=frame_stride,
        train=True
    )
    dataloader = torch.utils.data.DataLoader(adt_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if model_name == "simple_gaze_mlp":
        model = Simple_Eye_Gaze_MLP(input_dim=3 * len_per_input_seq // frame_stride, hidden_dim=hidden_dim, output_dim=3 * len_per_output_seq // frame_stride).to(device)
        criterion = Simple_Eye_Gaze_Loss()
        validation_criterion = Average_Gaze_Angular_Error()
    elif model_name == "simple_traj_mlp":
        model = Simple_Trajectory_MLP(input_dim=3 * len_per_input_seq // frame_stride + 16 * len_per_input_seq // frame_stride, hidden_dim=hidden_dim, output_dim=3 * len_per_output_seq // frame_stride + 4 * len_per_output_seq // frame_stride).to(device)
        criterion = Simple_Trajectory_Loss()
        validation_criterion = Average_Traj_Error()
    else:
        raise NotImplementedError


    if config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    
    writer = SummaryWriter()
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for idx, (input_clip, gt_clip) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training Epoch {epoch + 1}"):
            
            pred = model(input_clip, device)
            loss = criterion(pred, gt_clip, device)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            tqdm.write("loss: {:.4f}".format(loss.item()))
            running_loss += loss.item()
            if idx % 10 == 9:
                # print(f"Epoch: {epoch + 1}, Batch: {idx + 1}, Loss: {running_loss / 50}")
                writer.add_scalar('training loss', running_loss / 10, epoch * len(dataloader) + idx)
                running_loss = 0.0

        # validate
        validate(dataloader, model, model_name, criterion, validation_criterion, writer, epoch, device)
        
    print("Finish Training.")
    writer.close()

    torch.save({"model": model.state_dict()}, './logs/' + model_name + '.pth')
    print("Model saved.")

def validate(dataloader, model, model_name, criterion, validation_criterion, writer, epoch, device):
    model.eval()
    epoch_loss = 0
    if model_name == "simple_gaze_mlp":
        epoch_angular_error = 0
    if model_name == "simple_traj_mlp":
        epoch_coord_error = 0
        epoch_angular_error = 0

    with torch.no_grad():
        for idx, (input_clip, gt_clip) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Validation"):
            pred = model(input_clip, device)
            loss = criterion(pred, gt_clip, device)
            validation = validation_criterion(pred, gt_clip, device)

            epoch_loss += loss.item()
            
            if model_name == "simple_gaze_mlp":
                epoch_angular_error += validation.item()
            elif model_name == "simple_traj_mlp":
                epoch_coord_error += validation[0].item()
                epoch_angular_error += validation[1].item()

        
    writer.add_scalar('validation loss', epoch_loss / len(dataloader), epoch)
    if model_name == "simple_gaze_mlp":
        writer.add_scalar('validation angular error', epoch_angular_error / len(dataloader), epoch)
    elif model_name == "simple_traj_mlp":
        writer.add_scalar('validation coord error', epoch_coord_error / len(dataloader), epoch)
        writer.add_scalar('validation angular error', epoch_angular_error / len(dataloader), epoch)

if __name__ == "__main__":
    train()


