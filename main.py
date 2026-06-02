# coding=utf-8
import argparse
import io
from pathlib import Path
import os, sys, random
from models.DynamiTS import DynamiTS
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # main root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
import time
from utils.general import set_seed
from utils.dataloader import CustomDataLoader
import torch.nn.functional as F

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
class FlexibleFreDFLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mse = nn.MSELoss()
    def forward(self, outputs, batch_y):
        loss = 0
        loss_dict = {}
        if self.args.rec_lambda > 0:
            loss_rec = self.mse(outputs, batch_y)
            loss += self.args.rec_lambda * loss_rec
            loss_dict['loss_rec'] = loss_rec.item()
        if self.args.auxi_lambda > 0:
            loss_auxi = self._compute_auxiliary_loss(outputs, batch_y)
            loss += self.args.auxi_lambda * loss_auxi
            loss_dict['loss_auxi'] = loss_auxi.item()
        return loss, loss_dict

    def _compute_auxiliary_loss(self, outputs, batch_y):
        try:
            if self.args.auxi_mode == "fft":
                loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)
            elif self.args.auxi_mode == "rfft":
                if self.args.auxi_type == 'complex':
                    loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)
                elif self.args.auxi_type == 'complex-phase':
                    loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                elif self.args.auxi_type == 'complex-mag-phase':
                    loss_auxi_mag = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).abs()
                    loss_auxi_phase = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)).angle()
                    loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                elif self.args.auxi_type == 'phase':
                    loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                elif self.args.auxi_type == 'mag':
                    loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                elif self.args.auxi_type == 'mag-phase':
                    loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y, dim=1).abs()
                    loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y, dim=1).angle()
                    loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                else:
                    raise NotImplementedError
            elif self.args.auxi_mode == "rfft-D":
                loss_auxi = torch.fft.rfft(outputs, dim=-1) - torch.fft.rfft(batch_y, dim=-1)
            elif self.args.auxi_mode == "rfft-2D":
                loss_auxi = torch.fft.rfft2(outputs) - torch.fft.rfft2(batch_y)
            else:
                raise NotImplementedError
            if self.args.auxi_loss == "MAE":
                loss_auxi = loss_auxi.abs().mean() if getattr(self.args, 'module_first',
                                                              True) else loss_auxi.mean().abs()
            elif self.args.auxi_loss == "MSE":
                loss_auxi = (loss_auxi.abs() ** 2).mean() if getattr(self.args, 'module_first', True) else (
                            loss_auxi ** 2).mean().abs()
            else:
                raise NotImplementedError
            return loss_auxi
        except Exception as e:
            return F.mse_loss(outputs, batch_y)

class SafeAdvancedFreDFLoss(nn.Module):
    def __init__(self, rec_lambda=1.0, auxi_lambda=0.3, auxi_mode='rfft', auxi_type='complex', auxi_loss='MAE',
                 module_first=True):
        super().__init__()
        self.rec_lambda = rec_lambda
        self.auxi_lambda = auxi_lambda
        self.auxi_mode = auxi_mode
        self.auxi_type = auxi_type
        self.auxi_loss = auxi_loss
        self.module_first = module_first
        self.mse = nn.MSELoss()
    def compute_freq_loss(self, outputs, targets):
        try:
            if self.auxi_mode == "rfft":
                pred_fft = torch.fft.rfft(outputs, dim=-1)
                target_fft = torch.fft.rfft(targets, dim=-1)
                if self.auxi_type == 'complex':
                    loss_auxi = pred_fft - target_fft
                elif self.auxi_type == 'mag':
                    loss_auxi = pred_fft.abs() - target_fft.abs()
                elif self.auxi_type == 'phase':
                    loss_auxi = pred_fft.angle() - target_fft.angle()
                else:
                    loss_auxi = pred_fft - target_fft
            elif self.auxi_mode == "fft":
                loss_auxi = torch.fft.fft(outputs, dim=-1) - torch.fft.fft(targets, dim=-1)
            else:
                loss_auxi = outputs - targets

            if loss_auxi.numel() == 0:
                return torch.tensor(0.0, device=outputs.device, requires_grad=True)

            if self.auxi_loss == "MAE":
                freq_loss = loss_auxi.abs().mean()
            else:  # MSE
                freq_loss = (loss_auxi.abs() ** 2).mean()
            return freq_loss

        except Exception as e:
            return F.mse_loss(outputs, targets)

    def forward(self, outputs, targets):
        total_loss = 0
        loss_dict = {}
        if self.rec_lambda > 0:
            loss_rec = self.mse(outputs, targets)
            total_loss += self.rec_lambda * loss_rec
            loss_dict['rec_loss'] = float(loss_rec.item())
        if self.auxi_lambda > 0:
            loss_auxi = self.compute_freq_loss(outputs, targets)
            total_loss += self.auxi_lambda * loss_auxi
            loss_dict['auxi_loss'] = float(loss_auxi.item())
        return total_loss, loss_dict

class AdaptiveFreDFLoss(nn.Module):
    def __init__(self,
                 rec_lambda_start=1.0, rec_lambda_end=0.7,
                 auxi_lambda_start=0.1, auxi_lambda_end=0.5,
                 total_epochs=100, **kwargs):
        super().__init__()
        self.rec_lambda_start = rec_lambda_start
        self.rec_lambda_end = rec_lambda_end
        self.auxi_lambda_start = auxi_lambda_start
        self.auxi_lambda_end = auxi_lambda_end
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.base_loss = SafeAdvancedFreDFLoss(**kwargs)


    def set_epoch(self, epoch):
        self.current_epoch = epoch
        progress = epoch / self.total_epochs
        self.base_loss.rec_lambda = self.rec_lambda_start + \
                                    (self.rec_lambda_end - self.rec_lambda_start) * progress
        self.base_loss.auxi_lambda = self.auxi_lambda_start + \
                                     (self.auxi_lambda_end - self.auxi_lambda_start) * progress
    def forward(self, outputs, targets):
        return self.base_loss(outputs, targets)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main(args):
    # select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # workers
    torch.set_num_threads(4)
    # set seed
    set_seed(args.seed)

    # load datasets
    data_loader = CustomDataLoader(
        args.data,
        args.batch_size,
        args.seq_len,
        args.pred_len,
        args.feature_type,
        args.target,
    )
    train_data = data_loader.get_train()
    val_data = data_loader.get_val()
    test_data = data_loader.get_test()
    # load model
    model = DynamiTS(
        input_shape=(args.seq_len, data_loader.n_feature),
        pred_len=args.pred_len,
        dropout=args.dropout,
        n_block=args.n_block,
        patch=args.patch,
        k=args.down_sampling_layers,
        c=args.down_sampling_window,
        # stride=args.stride,
        alpha=args.alpha,
        target_slice=data_loader.target_slice,
        norm=args.norm,
        channel_wise=args.channel_wise,
        # layernorm=args.layernorm
    ).to(device)

    # criterion = torch.nn.MSELoss()
    if args.use_adaptive_fredf:
        criterion = AdaptiveFreDFLoss(
            rec_lambda_start=1.0, rec_lambda_end=0.7,
            auxi_lambda_start=0.1, auxi_lambda_end=0.5,
            total_epochs=args.train_epochs,
            auxi_mode=args.auxi_mode,
            auxi_type=args.auxi_type,
            auxi_loss=args.auxi_loss
        )
    else:
        criterion = FlexibleFreDFLoss(
            args
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-9)
    best_loss = torch.tensor(float('inf'))
    # create checkpoint directory
    save_directory = os.path.join(args.checkpoint_dir, args.name)
    if os.path.exists(save_directory):
        import glob
        import re
        path = Path(save_directory)
        dirs = glob.glob(f"{path}*")  # similar paths
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        save_directory = f"{path}{n}"  # update path
    os.makedirs(save_directory)
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(train_data):
            if i == 0:  # 只取第一个 batch
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, moe_loss, pre_weights = model(batch_x)
                break

    # start training
    for epoch in range(args.train_epochs):
        model.train()
        train_mloss = torch.zeros(1, device=device)
        iter_time = 0
        print(f"epoch : {epoch + 1}")
        print("Train")
        sync_cuda()
        epoch_start = time.perf_counter()
        pbar = tqdm(enumerate(train_data), total=len(train_data))
        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            start_time = time.time()
            sync_cuda()
            batch_start = time.perf_counter()
            outputs, moe_loss,weights = model(batch_x)
            main_loss, loss_dict = criterion(outputs, batch_y)
            optimizer.zero_grad()
                    # loss = main_loss + moe_loss + args.rate * contrastive_loss
            loss = main_loss + moe_loss
            loss.backward()
            optimizer.step()
            sync_cuda()
            batch_end = time.perf_counter()
            end_time = time.time()
            train_mloss = (train_mloss * i + loss.detach()) / (i + 1)
            iteration_time = (batch_end - batch_start) * 1000  # ms / batch
            iter_time = (iter_time * i + iteration_time) / (i + 1)
            pbar.set_description(('%-10s' * 1 + '%-10.8g ' * 1) % (f'{epoch + 1}/{args.train_epochs}', train_mloss))



        sync_cuda()
        epoch_end = time.perf_counter()
        epoch_train_time = epoch_end - epoch_start  # seconds
        print(f"train loss: {train_mloss.item()}, iter_time: {iter_time}," f"train batch time: {iter_time:.3f} ms/batch,"
        f"training time for one epoch: {epoch_train_time:.3f} s")


        model.eval()
        val_mloss = torch.zeros(1, device=device)
        val_mae = torch.zeros(1, device=device)
        val_mse = torch.zeros(1, device=device)
        print("Val")
        pbar = tqdm(enumerate(val_data), total=len(val_data))
        with torch.no_grad():
            for i, (batch_x, batch_y) in pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, moe_loss,weights= model(batch_x)
                criterion_result = criterion(outputs, batch_y)
                if isinstance(criterion_result, tuple):
                    loss, _ = criterion_result
                else:
                    loss = criterion_result
                val_mloss = (val_mloss * i + loss.detach()) / (i + 1)
                mae = torch.abs(outputs - batch_y).mean()
                val_mae = (val_mae * i + mae.detach()) / (i + 1)
                mse = ((outputs - batch_y) ** 2).mean()
                val_mse = (val_mse * i + mse.detach()) / (i + 1)
                pbar.set_description(('%-10s' * 1 + '%-10.8g' * 1) % (f'', val_mloss))

            if val_mloss < best_loss or epoch == args.train_epochs - 1:
                best_loss = val_mloss
                best_model = deepcopy(model.state_dict())
                torch.save(best_model, os.path.join(save_directory, "best.pt"))
        print(f"val loss: {val_mloss.item()}, val MSE: {val_mse.item()}, val MAE: {val_mae.item()}")

    # load best model
    model.load_state_dict(best_model)

    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(train_data):
            if i == 0:  # 只取第一个 batch
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs, moe_loss, post_weights = model(batch_x)

                break

    # start testing
    model.eval()

    test_mloss = torch.zeros(1, device=device)
    test_mae = torch.zeros(1, device=device)
    test_mse = torch.zeros(1, device=device)
    test_mape = torch.zeros(1, device=device)
    test_sse = torch.zeros(1, device=device)  # sum((pred - true)^2)
    test_true_sum = torch.zeros(1, device=device)  # sum(true)
    test_true_sq_sum = torch.zeros(1, device=device)  # sum(true^2)
    test_true_numel = torch.zeros(1, device=device)
    eps = 1e-5
    print("Final Test")
    pbar = tqdm(enumerate(test_data), total=len(test_data))
    inference_time_list = []  # ms / batch
    with torch.no_grad():
        for i, (batch_x, batch_y) in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            sync_cuda()
            infer_start = time.perf_counter()
            outputs, moe_loss ,weights= model(batch_x)
            sync_cuda()
            infer_end = time.perf_counter()
            infer_time = (infer_end - infer_start) * 1000  # ms / batch
            inference_time_list.append(infer_time)
            criterion_result = criterion(outputs, batch_y)
            if isinstance(criterion_result, tuple):
                loss, _ = criterion_result
            else:
                loss = criterion_result

            # loss = criterion(outputs, batch_y,return_dict=False)
            test_mloss = (test_mloss * i + loss.detach()) / (i + 1)
            mae = torch.abs(outputs - batch_y).mean()
            test_mae = (test_mae * i + mae.detach()) / (i + 1)
            mse = ((outputs - batch_y) ** 2).mean()
            test_mse = (test_mse * i + mse.detach()) / (i + 1)

            mape = torch.abs((outputs - batch_y) / (batch_y.abs() + eps)).mean()
            test_mape = (test_mape * i + mape.detach()) / (i + 1)

            test_rmse = torch.sqrt(test_mse)

            # RSE
            error = outputs - batch_y

            test_sse += torch.sum(error ** 2).detach()
            test_true_sum += torch.sum(batch_y).detach()
            test_true_sq_sum += torch.sum(batch_y ** 2).detach()
            test_true_numel += torch.tensor(batch_y.numel(), device=device, dtype=torch.float32)

            test_sst = test_true_sq_sum - (test_true_sum ** 2) / test_true_numel
            test_rse = torch.sqrt(test_sse / (test_sst + eps))


            pbar.set_description(('%-10.8g' * 1) % (test_mloss))

            pred = outputs.detach().cpu().numpy()  # .squeeze()
            true = batch_y.detach().cpu().numpy()  # .squeeze()

            root = Path(__file__).resolve().parent  # 当前脚本所在目录
            folder_path = root / 'test_results' / 'Traffic_512_192_fit_curves_normal_scripts'
            os.makedirs(folder_path, exist_ok=True)
            # if i % 20 == 0:
            #     input = batch_x.detach().cpu().numpy()
            #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
            #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
            #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

    # print(f"test loss: {test_mloss.item()}, test MSE: {test_mse.item()}, test MAE: {test_mae.item()}, test MAPE: {test_mape.item()}, test RMSE: {test_rmse.item()},test RSE: {test_rse.item()}")
    print(f"test loss: {test_mloss.item()}, test MSE: {test_mse.item()}, test MAE: {test_mae.item()}")



def infer_extension(dataset_name):
    if dataset_name.startswith('solar'):
        extension = 'txt'
    elif dataset_name.startswith('PEMS'):
        extension = 'npz'
    else:
        extension = 'csv'
    return extension


def parse_args():
    dataset = "ETTh1"
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    # data loader
    parser.add_argument('--data',
                        type=str,
                        default=ROOT / f'../data/{dataset}.{infer_extension(dataset)}',
                        help='dataset path')
    parser.add_argument(
        '--feature_type',
        type=str,
        default='M',
        choices=['S', 'M', 'MS'],
        help=(
            'forecasting task, options:[M, S, MS]; M:multivariate predict'
            ' multivariate, S:univariate predict univariate, MS:multivariate'
            ' predict univariate'
        ),
    )
    parser.add_argument(
        '--target', type=str, default='OT', help='target feature in S or MS task'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=ROOT / 'checkpoints',
        help='location of model checkpoints',
    )
    parser.add_argument(
        '--name',
        type=str,
        default=f'{dataset}',
        help='save best model to checkpoints/name',
    )
    # forecasting task
    parser.add_argument(
        # 96 192 336 512 672 720
        '--seq_len', type=int, default=720, help='input sequence length'
    )
    parser.add_argument(
        # 12 for PEMS  , {96, 192, 336, 720} for others
        '--pred_len', type=int, default=96, help='prediction sequence length'
    )
    # model hyperparameter
    parser.add_argument(
        # 1 2 3
        '--n_block',
        type=int,
        default=1,
        help='number of block for deep architecture',
    )
    parser.add_argument(
        # 0.0  0.5  1.0
        '--alpha',
        type=float,
        default=0.0,
        help='feature feature dimension',
    )
    parser.add_argument(
        '--down_sampling_layers',
        type=int,
        default=3,
        help='num of down sampling layers'
    )
    parser.add_argument(
        '--down_sampling_window',
        type=int,
        default=2,
        help='down sampling window size'
    )
    parser.add_argument(
        # 4 8 16
        '--patch',
        type=int,
        default=16,
        help='fully-connected history len',
    )
    parser.add_argument(
        '--norm',
        type=bool,
        default=True,
        help='RevIN',
    )
    parser.add_argument(
        '--layernorm',
        type=bool,
        default=True,
        help='layernorm',
    )
    parser.add_argument(
        '--channel_wise',
        type=bool,
        default=False,
        help='channel_wise',
    )
    parser.add_argument(
        '--dropout', type=float, default=0.1, help='dropout rate'
    )
    # optimization
    parser.add_argument(
        '--train_epochs', type=int, default=10, help='train epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=128, help='batch size of input data'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.00005,
        help='optimizer learning rate',
    )
    parser.add_argument(
        '--rate',
        type=float,
        default=1.0,
        help='optimizer learning rate',
    )
    parser.add_argument('--use_adaptive_fredf', action='store_true', help='use adaptive FreDF loss')
    parser.add_argument('--rec_lambda', type=float, default=1.0, help='reconstruction loss weight')
    parser.add_argument('--auxi_lambda', type=float, default=0.2, help='auxiliary frequency loss weight')
    parser.add_argument('--auxi_mode', type=str, default='rfft', choices=['rfft', 'fft','rfft-D', 'rfft-2D'], help='frequency domain mode')
    parser.add_argument('--auxi_type', type=str, default='complex',
                        choices=['complex', 'complex-phase', 'phase', 'mag', 'complex-mag-phase'],
                        help='frequency domain representation type')
    parser.add_argument('--auxi_loss', type=str, default='MAE', choices=['MAE', 'MSE'],
                        help='frequency domain loss function')

####
    parser.add_argument('--module_first', action='store_true', default=True,
                        help='apply abs() before mean() in loss calculation')
####
    # save results
    parser.add_argument(
        '--result_path', default='result.csv', help='path to save result'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("Args in experiment:")
    print(args)
    main(args)
