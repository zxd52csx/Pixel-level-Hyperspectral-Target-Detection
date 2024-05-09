import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.Dataset_spectra import HTD_dataset
from model.Siamese_fc import *
from model.evaluation import *
import tqdm
torch.set_default_tensor_type(torch.cuda.FloatTensor) 


def SFCTD(data_root, lr, batch_size, epoch_num, num_detectors, mixing_abundance, img_name, img_refer, prior_transform, divide, eo=False):
    train_dataset = HTD_dataset(data_root, img_name, img_refer=img_refer, eo=eo, prior_transform=prior_transform, divide=divide)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, generator=torch.Generator(device = 'cuda'))
    
    
    # initialization
    data_num = train_dataset.img.shape[0]
    band_num = train_dataset.img.shape[1]

    # model build
    detector = siamese_fc_cos_assemble(band_num, num_detectors=num_detectors).cuda()
    op = optim.Adam(detector.parameters(), lr=lr, weight_decay=0.0005)
    detector.train()

    # model training
    t1 = time.time()
    pbar = tqdm.tqdm(range(epoch_num), ncols=100)
    for _, e in enumerate(pbar):
        for _ in range(round(data_num / batch_size) + 1):
            losses = 0
            prior_list = []
            siamese_list = []
            for _ in range(num_detectors):
                prior, back = next(iter(train_dataloader))
                abundance = mixing_abundance
                pseudo_targets = (1 - abundance) * prior + abundance * back
                prior_branch = torch.cat([prior, prior])
                siamese_branch = torch.cat([back, pseudo_targets])
                prior_list.append(prior_branch)
                siamese_list.append(siamese_branch)
            prior_branch = torch.cat(prior_list,dim=1)
            siamese_branch = torch.cat(siamese_list,dim=1)
            label = torch.cat([torch.zeros(prior.shape[0]),torch.ones(prior.shape[0])]).to(siamese_branch.device)
            corrs, _ = detector([siamese_branch, prior_branch])
            for k in range(num_detectors):
                loss = detector.loss(label, corrs[:,k])
                losses += loss
            op.zero_grad()
            losses.backward()
            op.step()
        pbar.set_description(f"Epoch {e}/{epoch_num}")
        pbar.set_postfix({"loss:":loss.item()})
    print('SFCTD time comsumption:{}.'.format(time.time()-t1))
    
    # model testing
    detector.eval()
    test_img = train_dataset.img
    prior_ref = (torch.Tensor(train_dataset.prior[np.newaxis]) * torch.Tensor(test_img).norm(2, dim=1)[:, None]).cuda()
    t1 = time.time()
    Detection_result, _ = detector([torch.Tensor(test_img).cuda(), prior_ref], train=False)

    print('time comsumption:{}.'.format(time.time()-t1))
    Detection_result = Detection_result.mean(axis=-1).detach().cpu().numpy()
    row, col = train_dataset.groundtruth.shape

    print('*--------- {} -----------*'.format(img_name))
    evaluate_result = [Detection_result]
    evaluate_name = ['SFCTD']
    if eo:
        evaluate_result = evaluate_result + train_dataset.classic_results[0]
        evaluate_name = evaluate_name + train_dataset.classic_results[1]
    AUC_list = ROC(train_dataset.groundtruth.reshape(-1, order='F'),
        evaluate_result, 
        evaluate_name,
        img_name, row, col, eo)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("SFCTD")
    parser.add_argument('--cross_scene', action='store_true', help='Use cross-scene detection')
    parser.add_argument('--eo', action='store_true',help='evaluation other comparison')
    parser.add_argument('--batchsize', type=int, default=256, help='batch size')
    parser.add_argument('--ensemble_num', type=int, default=1, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='training epoch number')

    args = parser.parse_args()
    print(args)
    # important hyper-parameters
    mixing_abundance = 0.05

    # model ensemble
    num_detectors = args.ensemble_num

    # optimization hyper-parameters
    lr = 0.0001
    batch_size = args.batchsize
    Training_epoch = args.epoch
    # dataset dir
    data_root = 'data/MT-ABU-dataset'

    # detection type: same scene or cross_scene
    cross_scene = args.cross_scene
    if cross_scene:
        data_root = 'data/MT-ABU-dataset'
        # subset division is not used for cross scene detection
        args.SD = False
        # Inscene HSI
        img_list = ['A1', 'A2', 'B1', 'B2', 'U1', 'U2']
        # HSI to generate prior spectra
        img_refer_list = ['A2', 'A1', 'B2', 'B1', 'U2', 'U1']
        transform = 'MT'
    else:
        data_root = 'data/ABU-dataset'
        # Inscene HSI
        img_list = ['airport1', 'airport2', 'airport3', 'airport4', 'beach1', 'beach2', 'urban1', 'urban2']
        transform = 'part'

    for index, img_name in enumerate(img_list):
        if cross_scene:
            img_refer = [img_refer_list[index]]
        else:
            img_refer = [img_list[index]]
        SFCTD(data_root, lr, batch_size, Training_epoch, num_detectors, mixing_abundance, img_name, img_refer=img_refer, prior_transform=[transform], divide=1, eo=args.eo)

                    