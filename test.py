import torch, argparse
from base.dataset import ADT_Dataset
from model.simple import Simple_Eye_Gaze_MLP, Simple_Eye_Gaze_Loss
from base.utils import load_config
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from base.metrics import Average_Gaze_Angular_Error

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

    if model_name == "simple_mlp":
        model = Simple_Eye_Gaze_MLP(input_dim=3 * len_per_input_seq // frame_stride, hidden_dim=hidden_dim, output_dim=3 * len_per_output_seq // frame_stride).to(device)
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

    model.eval()
    sum_angular_error = 0
    with torch.no_grad():
        for idx, (input_clip, gt_clip) in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_eye_gaze = input_clip['eye_gaze'].to(device)
            gt_eye_gaze = gt_clip['eye_gaze'].to(device)

            pred_eye_gaze = model(input_eye_gaze)

            angular_error = Average_Gaze_Angular_Error(pred_eye_gaze, gt_eye_gaze)
            tqdm.write("Angular Error: {:.2f}°".format(angular_error))
            sum_angular_error += angular_error

    print("The average angular error is {:.2f}°.".format(sum_angular_error / len(dataloader)))
    
if __name__ == '__main__':
    test()
