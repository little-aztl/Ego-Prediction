import torch, argparse
from base.dataset import ADT_Dataset
from model.simple import Simple_Eye_Gaze_MLP, Simple_Eye_Gaze_Loss, Simple_Trajectory_Loss, Simple_Trajectory_MLP
from base.utils import load_config
from tqdm import tqdm
from train import validate
from base.metrics import Average_Gaze_Angular_Error, Average_Traj_Error

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path", type=str, default="./logs/simple_mlp.pth")
    args = parser.parse_args()

    config = load_config("./configs/config.yaml")
    model_name = config['model']
    len_per_input_seq = config['len_per_input_seq']
    len_per_output_seq = config['len_per_output_seq']
    interval = config['interval']
    frame_stride = config['frame_stride']
    hidden_dim = config['hidden_dim']
    use_gpu = config['use_gpu']
    num_workers = config['num_workers']

    if use_gpu:
        assert torch.cuda.is_available()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using device: ", device)

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

    model.load_state_dict(torch.load(args.model_path)['model'])

    dataset = ADT_Dataset(
        len_per_input_seq=len_per_input_seq, 
        len_per_output_seq=len_per_output_seq, 
        interval=interval, 
        stride=frame_stride,
        train=False
    )
    torch.multiprocessing.set_sharing_strategy('file_system')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    validate(dataloader, model, model_name, criterion, validation_criterion, None, 0, device, log_tensorboard=False)
    
if __name__ == '__main__':
    test()
