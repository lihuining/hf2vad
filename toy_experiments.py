import torch
import torchvision.transforms as transforms
from torchvision import datasets
import torch.nn as nn
import yaml
import shutil
from tqdm import tqdm
import os
import torch
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter

from datasets.dataset import img_batch_tensor2numpy, Chunked_sample_dataset
from models.ml_memAE_sc import ML_MemAE_SC
from utils.initialization_utils import weights_init_kaiming
from utils.model_utils import loader, saver, only_model_saver
from utils.vis_utils import visualize_sequences
import ml_memAE_sc_eval
# 1.定义超参数
BATCH_SIZE = 16

# 2.构建pipeline，对图片做一些变换
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),
    transforms.Resize([32,32]) # [h,w]
])

from torch.utils.data import DataLoader
import time


def train(config,config_path):
    paths = dict(log_dir="%s/%s" % (config["log_root"], config["exp_name"]+time.strftime("_%b%d_%H%M")),
                 ckpt_dir="%s/%s" % (config["ckpt_root"], config["exp_name"]))
    if not os.path.exists(paths["ckpt_dir"]):
        os.makedirs(paths["ckpt_dir"])
    if not os.path.exists(paths["log_dir"]):
        os.makedirs(paths["log_dir"])

    batch_size = config["batchsize"] # 256
    epochs = config["num_epochs"] # 80
    num_workers = config["num_workers"] # 8
    device = config["device"]

    mse_loss = nn.MSELoss().to(device)
    model = ML_MemAE_SC(num_in_ch=config["model_paras"]["motion_channels"],
                        seq_len=config["model_paras"]["num_flows"],
                        features_root=config["model_paras"]["feature_root"],
                        num_slots=config["model_paras"]["num_slots"],
                        shrink_thres=config["model_paras"]["shrink_thres"],
                        mem_usage=config["model_paras"]["mem_usage"],
                        skip_ops=config["model_paras"]["skip_ops"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], eps=1e-7, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.8)

    step = 0
    epoch_last = 0

    if not config["pretrained"]:
        model.apply(weights_init_kaiming)
    else:
        assert (config["pretrained"] is not None)
        model_state_dict, optimizer_state_dict, step = loader(config["pretrained"])
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        epoch_last = int(config["pretrained"].split('-')[-1])
        print('pretrained models loaded!', epoch_last)

    writer = SummaryWriter(paths["log_dir"])
    # copy config file
    shutil.copyfile(config_path,
                    os.path.join(config["log_root"], config["exp_name"], "ml_memAE_sc_cfg.yaml"))

    # Training
    best_auc = -1
    # dataset

    train_set = datasets.MNIST("/home/allenyljiang/Desktop/d2l-pytorch/data", train=True, download=False,
                               transform=pipeline)
    test_set = datasets.MNIST("/home/allenyljiang/Desktop/d2l-pytorch/data", train=False, download=False,
                              transform=pipeline)
    train_label_indices = [2]
    test_label_indices = [3]
    subset_testset = torch.utils.data.Subset(test_set,[i for i in range(len(test_set)) if test_set[i][1] in test_label_indices])
    test_dataloader = DataLoader(subset_testset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    for epoch in range(epoch_last, epochs + epoch_last):
    #for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):

        #dataset = Chunked_sample_dataset(os.path.join(training_chunked_samples_dir, chunk_file), last_flow=True)
        #dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


        # 选择特定标签的数据
        subset_trainset = torch.utils.data.Subset(train_set,[i for i in range(len(train_set)) if train_set[i][1] in train_label_indices])
        dataloader = DataLoader(subset_trainset, batch_size=batch_size,num_workers=num_workers, shuffle=True)

        for idx, train_data in tqdm(enumerate(dataloader),
                                    desc="Training Epoch %d" % (epoch + 1),
                                    total=len(dataloader)):
            model.train()

            sample_ofs, label = train_data # sample_ofs：digits ，（256,1,28,28）
            sample_ofs = sample_ofs.to(device)

            out = model(sample_ofs)
            loss_recon = mse_loss(out["recon"], sample_ofs)
            loss_sparsity = (
                    torch.mean(torch.sum(-out["att_weight3"] * torch.log(out["att_weight3"] + 1e-12), dim=1))
                    + torch.mean(torch.sum(-out["att_weight2"] * torch.log(out["att_weight2"] + 1e-12), dim=1))
                    + torch.mean(torch.sum(-out["att_weight1"] * torch.log(out["att_weight1"] + 1e-12), dim=1))
            )

            loss_all = config["lam_recon"] * loss_recon + config["lam_sparse"] * loss_sparsity

            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            if step % config["logevery"] == config["logevery"] - 1:
                print("[Step: {}/ Epoch: {}]: Loss: {:.4f} ".format(step + 1, epoch + 1, loss_all))

                writer.add_scalar('loss_total/train', loss_all, global_step=step + 1)
                writer.add_scalar('loss_recon/train', config["lam_recon"] * loss_recon, global_step=step + 1)
                writer.add_scalar('loss_sparsity/train', config["lam_sparse"] * loss_sparsity, global_step=step + 1)

                num_vis = 6
                writer.add_figure("img/train_sample_ofs",
                                  visualize_sequences(
                                      img_batch_tensor2numpy(sample_ofs.cpu()[:num_vis, :, :, :]),
                                      seq_len=sample_ofs.size(1),
                                      return_fig=True),
                                  global_step=step + 1)
                writer.add_figure("img/train_output",
                                  visualize_sequences(img_batch_tensor2numpy(
                                      out["recon"].detach().cpu()[:num_vis, :, :, :]),
                                      seq_len=out["recon"].size(1),
                                      return_fig=True),
                                  global_step=step + 1)
                writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=step + 1)

            step += 1
        del train_set

        scheduler.step()

        num_vis = 6
        if epoch % config["saveevery"] == config["saveevery"] - 1:
            model_save_path = os.path.join(paths["ckpt_dir"], config["model_savename"])
            saver(model.state_dict(), optimizer.state_dict(), model_save_path, epoch + 1, step, max_to_save=5)

            # evaluation 此处修改为在数字3上面进行evaluetae
            with torch.no_grad():
                test_features, test_labels = next(iter(test_dataloader))
                test_features = test_features.to(device)
                test_out = model(test_features)
                loss_recon = mse_loss(test_out["recon"], test_features)
                loss_sparsity = (
                        torch.mean(torch.sum(-test_out["att_weight3"] * torch.log(test_out["att_weight3"] + 1e-12), dim=1))
                        + torch.mean(torch.sum(-test_out["att_weight2"] * torch.log(test_out["att_weight2"] + 1e-12), dim=1))
                        + torch.mean(torch.sum(-test_out["att_weight1"] * torch.log(test_out["att_weight1"] + 1e-12), dim=1))
                )

                loss_all = config["lam_recon"] * loss_recon + config["lam_sparse"] * loss_sparsity
                writer.add_scalar('loss_total/test', loss_all, global_step=epoch + 1)
                writer.add_figure("img/test_input",
                                  visualize_sequences(img_batch_tensor2numpy(
                                      test_features.detach().cpu()[:num_vis, :, :, :]),
                                      seq_len=test_features.size(1),
                                      return_fig=True),
                                      global_step=epoch + 1)
                writer.add_figure("img/test_output",
                                  visualize_sequences(img_batch_tensor2numpy(
                                      test_out["recon"].detach().cpu()[:num_vis, :, :, :]),
                                      seq_len=test_out["recon"].size(1),
                                      return_fig=True),
                                      global_step=epoch + 1)


if __name__ == '__main__':
    config_path = "./cfgs/toy_ml_memAE_sc_cfg.yaml"
    config = yaml.safe_load(open(config_path))

    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]

    train(config,config_path)