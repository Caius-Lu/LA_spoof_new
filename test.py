import argparse
import torch
from tqdm import tqdm
import datasets.dataset as module_data
from datasets.dataset import *
import models.loss as module_loss
import models.metric as module_metric
import models.model as module_arch
from parse_config import ConfigParser
from torch.utils.data import DataLoader
from torch import nn
import os
def main(config):
    logger = config.get_logger('test')

    # setup datasets instances
    # # test dataset
    mode  = config.args.mode
    out_dir = config.args.output_dir
    #

    test_arg = config['dataset'][mode]
    test_set = getattr(module_data, test_arg['dataset']['type'])(**dict(test_arg['dataset']['args']))
    test_loader = DataLoader(test_set, **dict(test_arg['loader']))
    save_path = './scores'
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    if not os.path.exists(save_path):
        os.mkdirs(save_path)
    eval_output = os.path.join(save_path, out_dir)

    # loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    # prepare model for testing
    model = model.to(device)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)


    weight = torch.FloatTensor([1.0, 9.0]).to(device)
    loss_fn = nn.NLLLoss(weight=weight)

    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    fname_list = []
    sys_id_list = []
    key_list = []
    score_list = []

    with torch.no_grad():
        for i, (data, target,batch_meta) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device).long()
            output = model(data)
            batch_score = (output[:, 1] - output[:, 0]).data.cpu().numpy().ravel()
            # add outputs
            fname_list.extend(list(batch_meta[1]))
            key_list.extend(['bonafide' if key == 1 else 'spoof' for key in list(batch_meta[4])])
            sys_id_list.extend([test_set.sysid_dict_inv[s.item()] for s in list(batch_meta[3])])
            score_list.extend(batch_score.tolist())
            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
        with open(eval_output, 'w') as fh:
            for f, s, k, cm in zip(fname_list, sys_id_list, key_list, score_list):
                fh.write('{} {} {} {}\n'.format(f, s, k, cm))
        print('Result saved to {}'.format(save_path))

    n_samples = len(test_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default='/data3/luchao/LA_spoof_new/saved/models/ASVspoof2019_LA/0701_202606/model_best.pth', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-o', '--output_dir', default=None, type=str,
                      help='Path to save the evaluation result')
    args.add_argument('-m', '--mode', default='validate', type=str,
                      help='val or test')

    config = ConfigParser.from_args(args)
    main(config)
