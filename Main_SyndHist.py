import os
import random
import argparse
import json
import logging
import torch
from torch.utils import data
from datetime import datetime
from Codes import Get_Generator_and_Parity, sign_to_bin, bin_to_sign, EbN0_to_std
from torch.optim.lr_scheduler import CosineAnnealingLR
from Model import GRUSyndromeReduce
from torch.utils.data import DataLoader
import time

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def train(model, device, train_loader, optimizer, epoch, LR):
    model.train()
    t = time.time()
    cum_loss = 0
    for batch_idx, (final_synd, syndrome_hist, mask) in enumerate(
        train_loader
    ):  
        synd_pre = model(syndrome_hist.to(device), mask.to(device))
        loss = model.loss(-synd_pre, final_synd.to(device))
        cum_loss += loss.item() * final_synd.shape[0]
        model.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f"Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}"
            )
    logging.info(f"Epoch {epoch} Train Time {time.time() - t}s\n")
    return cum_loss

##################################################################


class SyndHistDataset(data.Dataset):
    def __init__(
        self,
        code,
        sigma,
        len,
        syndrome_sigma,
        readout_sigma,
        zero_cw=True,
        max_hist=None,
        equal_hist=False
    ):
        self.equal_hist = equal_hist
        self.max_hist = max_hist
        self.code = code
        self.sigma = sigma
        self.syndrome_sigma = syndrome_sigma
        self.readout_sigma = readout_sigma
        self.len = len
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)

        self.zero_word = torch.zeros((self.code.k)).long() if zero_cw else None
        self.zero_cw = torch.zeros((self.code.n)).long() if zero_cw else None
        self.m = code.pc_matrix.size(0)  # number of parity checks

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.zero_cw is None:
            m = torch.randint(0, 2, (1, self.code.k)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        if self.equal_hist:
            hist_len = self.max_hist
        else:
            hist_len = random.randint(1, self.max_hist)
        z = torch.randn(self.code.n) * random.choice(self.sigma)
        y = None
        # add purtubation
        hist = []
        for _ in range(hist_len):
            if y is None:
                y = bin_to_sign(x) + z
            syndrome_no_noise = (
                torch.matmul(sign_to_bin(torch.sign(y)).long(), self.pc_matrix) % 2
            )
            syndrome_no_noise = bin_to_sign(syndrome_no_noise)
            syndrome_sigma = random.choice(self.syndrome_sigma)
            if syndrome_sigma is None:
                syndrome_noise = torch.zeros(len(syndrome_no_noise))
            else:
                syndrome_noise = torch.randn(len(syndrome_no_noise)) * syndrome_sigma
            syndrome = syndrome_no_noise.float() + syndrome_noise
            # post_measurement readout noise
            readout_sigma = random.choice(self.readout_sigma)
            if readout_sigma is None:
                y_noise = torch.zeros(len(y))
            else:
                y_noise = torch.randn(len(y)) * readout_sigma
            y += y_noise
            hist.append(syndrome)

        # pad to max_hist and build mask
        if hist_len < self.max_hist:
            pad = [torch.zeros(self.m)] * (self.max_hist - hist_len)
            hist = hist + pad

        syndrome_hist = torch.stack(hist, dim=0)  # (T_max, m)
        mask = torch.tensor(
            [True] * hist_len + [False] * (self.max_hist - hist_len)
        )
        return (
            syndrome_no_noise.float(),  # (m,)
            syndrome_hist.float(),  # (T_max, m)
            mask.bool(),  # (T_max,)
        )



def main(args):
    code = args.code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUSyndromeReduce(code.pc_matrix.size(0)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logging.info(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logging.info(
        f"Total params: {total_params}, Trainable params: {trainable_params}"
    )
    #################################
    EbNo_range_test = [5]
    EbNo_range_train = range(2, 8)
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]

    ebno_syndrome_range = [None] + list(range(10, 31, 5))
    ebno_readout_range = [None] + list(range(10, 31, 5))
    syndrome_sigma_list = [
        EbN0_to_std(ebno, code.k / code.n) if ebno is not None else None
        for ebno in ebno_syndrome_range
    ]
    readout_sigma_list = [
        EbN0_to_std(ebno, code.k / code.n) if ebno is not None else None
        for ebno in ebno_readout_range
    ]

    ebno_syndrome_test_range = [None, 10]
    ebno_readout_test_range = [None, 10]

    train_dataloader = DataLoader(
        SyndHistDataset(
            code,
            std_train,
            len=args.batch_size * 100,
            syndrome_sigma=syndrome_sigma_list,
            readout_sigma=readout_sigma_list,
            zero_cw=True,
            max_hist=args.max_hist,
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=args.workers,
    )
    #################################
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        logging.info(f"Epoch {epoch}/{args.epochs}")
        loss = train(
            model,
            device,
            train_dataloader,
            optimizer,
            epoch,
            LR=scheduler.get_last_lr()[0],
        )
        logging.info(f"Epoch {epoch} Loss: {loss:.4f}")
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, "best_model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Syndrome history compressor")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpus", type=str, default="-1", help="gpus ids")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--max_hist", type=int, default=10)

    # Code args
    parser.add_argument(
        "--code_type",
        type=str,
        default="POLAR",
        choices=["BCH", "POLAR", "LDPC", "CCSDS", "MACKAY"],
    )
    parser.add_argument("--code_k", type=int, default=32)
    parser.add_argument("--code_n", type=int, default=64)

    args = parser.parse_args()

    og_args_dict = vars(args).copy()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)
    ####################################################################

    class Code:
        pass

    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    G, H = Get_Generator_and_Parity(code, standard_form=args.standardize)
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    args.code = code

    base_dir = "/home/rbali/Results_SyndCompress/"

    model_dir = os.path.join(
        base_dir,
        args.code_type
        + "__Code_n_"
        + str(args.code_n)
        + "_k_"
        + str(args.code_k)
        + "__"
        + datetime.now().strftime("%d_%m_%Y_%H_%M_%S"),
    )

    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir

    with open(os.path.join(args.path, "args.json"), "w") as f:
        json.dump(og_args_dict, f, indent=4)

    handlers = [logging.FileHandler(os.path.join(args.path, "logging.txt"))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=handlers
    )
    logging.info(f"Path to model/logs: {args.path}")
    logging.info(args)
    main(args)
