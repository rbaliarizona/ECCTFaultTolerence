"""
Implementation of "Error Correction Code Transformer" (ECCT)
https://arxiv.org/abs/2203.14966
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from __future__ import print_function
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
from Codes import *
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from Model import ECCTransformerWithSyndromeHistory
import tempfile
import pdb
import json
from torch.utils.tensorboard import SummaryWriter

##################################################################
##################################################################


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


##################################################################


class ECC_Dataset_w_History(data.Dataset):
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
        y = bin_to_sign(x) + z
        syndrome = (
            torch.matmul(sign_to_bin(torch.sign(y)).long(), self.pc_matrix) % 2
        )
        syndrome = bin_to_sign(syndrome)
        # add purtubation
        hist = []
        for _ in range(hist_len):
            syndrome_noise = torch.randn(len(syndrome)) * random.choice(
                self.syndrome_sigma
            )  # add noise to syndrome
            syndrome = syndrome.float() + syndrome_noise
            # post_measurement readout noise
            y_noise = torch.randn(len(y)) * random.choice(self.readout_sigma)
            y += y_noise
            hist.append(syndrome)
        magnitude = torch.abs(y)

        # pad to max_hist and build mask
        if hist_len < self.max_hist:
            pad = [torch.zeros(self.m)] * (self.max_hist - hist_len)
            hist = hist + pad

        syndrome_hist = torch.stack(hist, dim=0)  # (T_max, m)
        mask = torch.tensor(
            [True] * hist_len + [False] * (self.max_hist - hist_len)
        )
        return (
            m.float(),
            x.float(),
            z.float(),
            y.float(),
            magnitude.float(),
            syndrome_hist.float(),  # (T_max, m)
            mask.bool(),  # (T_max,)
        )


##################################################################
##################################################################


def train(model, device, train_loader, optimizer, epoch, LR):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = 0
    t = time.time()
    for batch_idx, (m, x, z, y, magnitude, syndrome_hist, mask) in enumerate(
        train_loader
    ):
        z_mul = y * bin_to_sign(x)
        z_pred = model(
            magnitude.to(device), syndrome_hist.to(device), mask.to(device)
        )
        loss, x_pred = model.loss(-z_pred, z_mul.to(device), y.to(device))
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        ber = BER(x_pred, x.to(device))
        fer = FER(x_pred, x.to(device))

        cum_loss += loss.item() * x.shape[0]
        cum_ber += ber * x.shape[0]
        cum_fer += fer * x.shape[0]
        cum_samples += x.shape[0]
        if (batch_idx + 1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f"Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}"
            )
    logging.info(f"Epoch {epoch} Train Time {time.time() - t}s\n")
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


##################################################################


def test(
    model, device, test_loader_list, std_to_ebno, min_FER=100
):
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = (
        [],
        [],
        [],
        [],
    )
    channel_ebno_list = []
    syndrome_ebno_list = []
    readout_ebno_list = []
    max_hist_list = []
    t = time.time()
    with torch.no_grad():
        for test_loader in test_loader_list:
            test_loss = test_ber = test_fer = cum_count = 0.0
            channel_ebno = std_to_ebno(test_loader.dataset.sigma[0])
            syndrome_ebno = std_to_ebno(test_loader.dataset.syndrome_sigma[0])
            readout_ebno = std_to_ebno(test_loader.dataset.readout_sigma[0])
            max_hist = test_loader.dataset.max_hist
            channel_ebno_list.append(channel_ebno)
            syndrome_ebno_list.append(syndrome_ebno)
            readout_ebno_list.append(readout_ebno)
            max_hist_list.append(max_hist)
            while True:
                (m, x, z, y, magnitude, syndrome_hist, mask) = next(
                    iter(test_loader)
                )
                z_mul = y * bin_to_sign(x)
                z_pred = model(
                    magnitude.to(device),
                    syndrome_hist.to(device),
                    mask.to(device),
                )
                loss, x_pred = model.loss(
                    -z_pred, z_mul.to(device), y.to(device)
                )

                test_loss += loss.item() * x.shape[0]
                test_ber += BER(x_pred, x.to(device)) * x.shape[0]
                test_fer += FER(x_pred, x.to(device)) * x.shape[0]
                cum_count += x.shape[0]
                if (
                    min_FER > 0 and test_fer > min_FER and cum_count > 1e4
                ) or cum_count >= 1e5:
                    if cum_count >= 1e5:
                        print(
                            f"Number of samples threshold reached for channel EbN0:{channel_ebno}, syndrome EbN0:{syndrome_ebno}, readout EbN0:{readout_ebno}, maxHist:{max_hist}"
                        )
                    else:
                        print(
                            f"FER count threshold reached for channel EbN0:{channel_ebno}, syndrome EbN0:{syndrome_ebno}, readout EbN0:{readout_ebno}, maxHist:{max_hist} (FER={test_fer / cum_count:.2e})"
                        )
                    break
            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)
            print(
                f"Test EbN0={channel_ebno}, Syndrome EbN0={syndrome_ebno}, Readout EbN0={readout_ebno}, BER={test_loss_ber_list[-1]:.2e}"
            )
        ###
        logging.info(
            "\nTest Loss "
            + " ".join(
                [
                    "\nNumHist: {}  EnB0s: {}, {}, {}; Value:{:.2e}".format(
                        max_hist, ebno, synd_ebno, rd_ebno, elem
                    )
                    for (elem, ebno, synd_ebno, rd_ebno, max_hist) in (
                        zip(
                            test_loss_list,
                            channel_ebno_list,
                            syndrome_ebno_list,
                            readout_ebno_list,
                            max_hist_list,
                        )
                    )
                ]
            )
        )
        logging.info(
            "\nTest FER "
            + " ".join(
                [
                    "\nNumHist: {}  EbN0s: {}, {}, {}; Value:{:.2e}".format(
                        max_hist, ebno, synd_ebno, rd_ebno, elem
                    )
                    for (elem, ebno, synd_ebno, rd_ebno, max_hist) in (
                        zip(
                            test_loss_fer_list,
                            channel_ebno_list,
                            syndrome_ebno_list,
                            readout_ebno_list,
                            max_hist_list,
                        )
                    )
                ]
            )
        )
        logging.info(
            "\nTest BER "
            + " ".join(
                [
                    "\nNumHist: {}  EbN0s: {}, {}, {}; Value:{:.2e}".format(
                        max_hist, ebno, synd_ebno, rd_ebno, elem
                    )
                    for (elem, ebno, synd_ebno, rd_ebno, max_hist) in (
                        zip(
                            test_loss_ber_list,
                            channel_ebno_list,
                            syndrome_ebno_list,
                            readout_ebno_list,
                            max_hist_list,
                        )
                    )
                ]
            )
        )
        logging.info(
            "\nTest -ln(BER) "
            + " ".join(
                [
                    "\nNumHist: {}  EbN0s: {}, {}, {}; Value:{:.2e}".format(
                        max_hist, ebno, synd_ebno, rd_ebno, -np.log(elem)
                    )
                    for (elem, ebno, synd_ebno, rd_ebno, max_hist) in (
                        zip(
                            test_loss_ber_list,
                            channel_ebno_list,
                            syndrome_ebno_list,
                            readout_ebno_list,
                            max_hist_list,
                        )
                    )
                ]
            )
        )
    logging.info(
        f"# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n"
    )
    return test_loss_list, test_loss_ber_list, test_loss_fer_list


##################################################################
##################################################################
##################################################################


def main(args):
    code = args.code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.path, "runs"))

    #################################
    model = ECCTransformerWithSyndromeHistory(args).to(device)

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
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]

    # Use range(10, 5, 21) for syndrome and readout sigma
    ebno_syndrome_range = [10]
    ebno_readout_range = [100]
    syndrome_sigma_list = [
        EbN0_to_std(ebno, code.k / code.n) for ebno in ebno_syndrome_range
    ]
    readout_sigma_list = [
        EbN0_to_std(ebno, code.k / code.n) for ebno in ebno_readout_range
    ]

    ebno_syndrome_test_range = [10]
    ebno_readout_test_range = [75]
    syndrome_sigma_test_list = [
        EbN0_to_std(ebno, code.k / code.n) for ebno in ebno_syndrome_test_range
    ]
    readout_sigma_test_list = [
        EbN0_to_std(ebno, code.k / code.n) for ebno in ebno_readout_test_range
    ]

    train_dataloader = DataLoader(
        ECC_Dataset_w_History(
            code,
            std_train,
            len=args.batch_size * 1000,
            syndrome_sigma=syndrome_sigma_list,
            readout_sigma=readout_sigma_list,
            zero_cw=True,
            max_hist=args.max_hist,
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=args.workers,
    )
    # Generate three equally spaced history lengths between 1 and max_hist (inclusive)
    history_lengths = [1, args.max_hist // 2, args.max_hist]
    test_dataloader_list = [
        DataLoader(
            ECC_Dataset_w_History(
                code,
                [std_test[ii]],
                len=int(args.test_batch_size),
                syndrome_sigma=[syndrome_sigma_test_list[jj]],
                readout_sigma=[readout_sigma_test_list[kk]],
                zero_cw=False,
                max_hist=hh,
                equal_hist=True,
            ),
            batch_size=int(args.test_batch_size),
            shuffle=False,
            num_workers=args.workers,
        )
        for ii in range(len(std_test))
        for jj in range(len(syndrome_sigma_test_list))
        for kk in range(len(readout_sigma_test_list))
        for hh in history_lengths
    ]

    std_to_ebno = lambda std: std_to_EbN0(std, code.k / code.n)  # noqa: E731
    #################################
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        logging.info(f"Epoch {epoch}/{args.epochs}")
        loss, ber, fer = train(
            model,
            device,
            train_dataloader,
            optimizer,
            epoch,
            LR=scheduler.get_last_lr()[0],
        )
        scheduler.step()

        # Log training loss, BER, FER
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("BER/train", ber, epoch)
        writer.add_scalar("FER/train", fer, epoch)

        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, "best_model"))
        if epoch % 25 == 0 or epoch in [1, args.epochs]:
            test_loss_list, test_ber_list, test_fer_list = test(
                model, device, test_dataloader_list, std_to_ebno
            )
            for ebno, loss, ber, fer in zip(
                EbNo_range_test, test_loss_list, test_ber_list, test_fer_list
            ):
                writer.add_scalar(f"Loss/test_EbN0_{ebno}", loss, epoch)
                writer.add_scalar(f"BER/test_EbN0_{ebno}", ber, epoch)
                writer.add_scalar(f"FER/test_EbN0_{ebno}", fer, epoch)
    writer.close()


##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch ECCT")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpus", type=str, default="-1", help="gpus ids")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--standardize", action="store_true")

    # model args
    parser.add_argument("--N_dec", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--h", type=int, default=8)
    # The model is that
    # y = x + noise (selected from multiple gaussians with different EbNo in code)
    # s = H y + noise_syndrome  (ebno_syndrome_dB is for this)
    # The user sees y' = y + noise_readout (ebno_readout_dB is for this)
    parser.add_argument(
        "--tmp",
        action="store_true",
        help="Stores outputs in a temporary directory. For debugging purposes.",
    )

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
    ####################################################################
    if args.tmp:
        temp_dir = tempfile.mkdtemp()
        args.path = temp_dir
        print(f"Temporary directory created at: {temp_dir}")
    else:
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        model_dir = os.path.join(
            f"/home/rbali/Results_{script_name}",
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
