import torch
from base.dataset import ADT_Dataset
from model.simple import Simple_Eye_Gaze_MLP, Simple_Eye_Gaze_Loss
from model.simple_transformer import Simple_Transformer_Eye_Gaze
from base.utils import load_config
from base.metrics import Average_Angular_Error
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def validate(epoch, model, data_loader, criterion, writer, device):
    model.eval()
    epoch_loss = 0
    epoch_angular_error = 0
    with torch.no_grad():
        for idx, (input_clip, gt_clip) in tqdm(enumerate(data_loader), total=len(data_loader)):
            input_eye_gaze = input_clip['eye_gaze'].to(device)
            gt_eye_gaze = gt_clip['eye_gaze'].to(device)

            pred_eye_gaze = model(input_eye_gaze)
            loss = criterion(pred_eye_gaze, gt_eye_gaze)
            angular_error = Average_Angular_Error(pred_eye_gaze, gt_eye_gaze)

            epoch_loss += loss.item()
            epoch_angular_error += angular_error

    
    writer.add_scalar('validation loss', epoch_loss / len(data_loader), epoch)
    writer.add_scalar('validation angular error', epoch_angular_error / len(data_loader), epoch)
            

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

    if model_name == "simple_mlp":
        model = Simple_Eye_Gaze_MLP(input_dim=3 * len_per_input_seq // frame_stride, hidden_dim=hidden_dim, output_dim=3 * len_per_output_seq // frame_stride).to(device)
    elif model_name == "simple_transformer":
        model = Simple_Transformer_Eye_Gaze().to(device)
    else:
        raise NotImplementedError

    criterion = Simple_Eye_Gaze_Loss()

    if config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    
    writer = SummaryWriter()
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for idx, (input_clip, gt_clip) in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_eye_gaze = input_clip['eye_gaze'].to(device)
            gt_eye_gaze = gt_clip['eye_gaze'].to(device)

            optimizer.zero_grad()
            pred_eye_gaze = model(input_eye_gaze)
            loss = criterion(pred_eye_gaze, gt_eye_gaze)
            loss.backward()
            optimizer.step()

            tqdm.write("loss: {:.4f}".format(loss.item()))
            running_loss += loss.item()
            if idx % 50 == 49:
                # print(f"Epoch: {epoch + 1}, Batch: {idx + 1}, Loss: {running_loss / 50}")
                writer.add_scalar('training loss', running_loss / 50, epoch * len(dataloader) + idx)
                running_loss = 0.0

        validate(epoch + 1, model, dataloader, criterion, writer, device)
        
    print("Finish Training.")
    writer.close()

    torch.save({"model": model.state_dict()}, './logs/' + model_name + '.pth')
    print("Model saved.")

if __name__ == "__main__":
    train()


