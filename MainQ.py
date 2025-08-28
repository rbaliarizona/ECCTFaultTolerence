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
from Model import (
    ECC_Transformer,
    NoisySyndromeECCTDecoder,
    NoisySyndromeRNNECCTDecoder,
)
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


class ECC_Dataset(data.Dataset):
    def __init__(self, code, sigma, len, zero_cw=True):
        self.code = code
        self.sigma = sigma
        self.len = len
        self.generator_matrix = code.generator_matrix.transpose(0, 1)
        self.pc_matrix = code.pc_matrix.transpose(0, 1)
        self.zero_word = torch.zeros((self.code.k)).long() if zero_cw else None
        self.zero_cw = torch.zeros((self.code.n)).long() if zero_cw else None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.zero_cw is None:
            m = torch.randint(0, 2, (1, self.code.k)).squeeze()
            x = torch.matmul(m, self.generator_matrix) % 2
        else:
            m = self.zero_word
            x = self.zero_cw
        z = torch.randn(self.code.n) * random.choice(self.sigma)
        y = bin_to_sign(x) + z
        return (
            m.float(),
            x.float(),
            z.float(),
            y.float(),
        )


##################################################################
##################################################################


def train(model, device, train_loader, optimizer, epoch, LR):
    model.train()
    cum_loss = cum_ber = cum_fer = cum_samples = 0
    stop_sum = 0.0
    stop_hist = torch.zeros(getattr(model, "max_steps", 1), dtype=torch.long)
    t = time.time()

    for batch_idx, (m, x, z, y) in enumerate(train_loader):
        y = y.to(device)
        x = x.to(device)

        y_final, z_pred, stop_step = model(y, x)  # stop_step: [B], 0-based
        loss, x_pred = model.loss(-z_pred, y_final, x)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics
        bsz = x.shape[0]
        ber = BER(x_pred, x)
        fer = FER(x_pred, x)

        cum_loss += loss.item() * bsz
        cum_ber += ber * bsz
        cum_fer += fer * bsz
        cum_samples += bsz

        ss = stop_step.cpu()
        stop_sum += ss.float().sum().item()
        # grow hist if needed (defensive)
        if ss.max().item() >= stop_hist.numel():
            new_len = int(max(ss.max().item() + 1, stop_hist.numel()))
            new_hist = torch.zeros(new_len, dtype=torch.long)
            new_hist[: stop_hist.numel()] = stop_hist
            stop_hist = new_hist
        counts = torch.bincount(ss, minlength=stop_hist.numel())
        stop_hist[: counts.numel()] += counts

        if (batch_idx + 1) % 500 == 0 or batch_idx == len(train_loader) - 1:
            avg_stop = stop_sum / max(1, cum_samples)  # 0-based
            logging.info(
                f"Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: "
                f"LR={LR:.2e}, Loss={cum_loss / cum_samples:.2e} "
                f"BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e} "
                f"AvgStop(0b)={avg_stop:.2f} Hist={stop_hist.tolist()}"
            )

    logging.info(f"Epoch {epoch} Train Time {time.time() - t:.2f}s\n")
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_fer / cum_samples


##################################################################


def test(model, device, test_loader_list, EbNo_range_test, min_FER=100):
    model.eval()
    test_loss_list, test_loss_ber_list, test_loss_fer_list, cum_samples_all = (
        [],
        [],
        [],
        [],
    )
    test_avg_stop_list, test_stop_hist_list = [], []
    t = time.time()

    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_loss = test_ber = test_fer = cum_count = 0.0
            stop_sum = 0.0
            stop_hist = None

            while True:
                (m, x, z, y) = next(iter(test_loader))
                y = y.to(device)
                x = x.to(device)

                y_final, z_pred, stop_step = model(
                    y, x
                )  # stop_step: [B], 0-based
                loss, x_pred = model.loss(-z_pred, y_final, x)

                bsz = x.shape[0]
                test_loss += loss.item() * bsz
                test_ber += BER(x_pred, x) * bsz
                test_fer += FER(x_pred, x) * bsz
                cum_count += bsz

                # ---- stopping stats
                ss = stop_step.detach().cpu()
                stop_sum += (
                    ss.float().sum().item()
                )  # still fine: mean of 0-based indices
                if stop_hist is None:
                    L = int(ss.max().item())
                    stop_hist = torch.zeros(
                        max(L, getattr(model, "max_steps", L)) + 1,
                        dtype=torch.long,
                    )
                counts = torch.bincount(ss, minlength=stop_hist.numel())
                stop_hist[: counts.numel()] += counts

                if (
                    min_FER > 0 and test_fer > min_FER and cum_count > 1e5
                ) or cum_count >= 1e9:
                    break

            cum_samples_all.append(cum_count)
            test_loss_list.append(test_loss / cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_fer_list.append(test_fer / cum_count)

            avg_stop = stop_sum / cum_count  # this is 0-based average
            test_avg_stop_list.append(avg_stop)

            # histogram: bins correspond to steps [0..max_steps-1]
            hist_readable = stop_hist[
                : getattr(model, "max_steps", stop_hist.numel())
            ].tolist()
            test_stop_hist_list.append(hist_readable)

            print(
                f"Test EbN0={EbNo_range_test[ii]}, BER={test_loss_ber_list[-1]:.2e}, "
                f"AvgStop={avg_stop:.2f}, Hist={hist_readable}"
            )

        # ----- logging summaries
        logging.info(
            "Test Loss "
            + " ".join(
                [
                    f"{ebno}: {elem:.2e}"
                    for elem, ebno in zip(test_loss_list, EbNo_range_test)
                ]
            )
        )
        logging.info(
            "Test FER "
            + " ".join(
                [
                    f"{ebno}: {elem:.2e}"
                    for elem, ebno in zip(test_loss_fer_list, EbNo_range_test)
                ]
            )
        )
        logging.info(
            "Test BER "
            + " ".join(
                [
                    f"{ebno}: {elem:.2e}"
                    for elem, ebno in zip(test_loss_ber_list, EbNo_range_test)
                ]
            )
        )
        logging.info(
            "Test -ln(BER) "
            + " ".join(
                [
                    f"{ebno}: {(-np.log(elem)):.2e}"
                    for elem, ebno in zip(test_loss_ber_list, EbNo_range_test)
                ]
            )
        )
        logging.info(
            "Test AvgStop "
            + " ".join(
                [
                    f"{ebno}: {avg:.2f}"
                    for avg, ebno in zip(test_avg_stop_list, EbNo_range_test)
                ]
            )
        )
        logging.info(
            "Test StopHist "
            + " ".join(
                [
                    f"{ebno}: {hist}"
                    for hist, ebno in zip(test_stop_hist_list, EbNo_range_test)
                ]
            )
        )

    logging.info(
        f"# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n"
    )
    return (
        test_loss_list,
        test_loss_ber_list,
        test_loss_fer_list,
    )


##################################################################
##################################################################
##################################################################


def main(args):
    code = args.code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.path, "runs"))
    #################################
    syndrome_sigma = EbN0_to_std(args.ebno_syndrome_dB, code.k / code.n)
    readout_sigma = EbN0_to_std(args.ebno_readout_dB, code.k / code.n)

    ecct_model = ECC_Transformer(args, dropout=0).to(device)
    model = NoisySyndromeRNNECCTDecoder(
        ecct_model,
        code.pc_matrix.to(device),
        syndrome_sigma=syndrome_sigma,
        readout_sigma=readout_sigma,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logging.info(model)
    logging.info(
        f"# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}"
    )
    #################################
    EbNo_range_test = range(4, 7)
    EbNo_range_train = range(2, 8)
    std_train = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_train]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    train_dataloader = DataLoader(
        ECC_Dataset(
            code,
            std_train,
            len=args.batch_size * 1000,
            zero_cw=True,
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=args.workers,
    )
    test_dataloader_list = [
        DataLoader(
            ECC_Dataset(
                code,
                [std_test[ii]],
                len=int(args.test_batch_size),
                zero_cw=False,
            ),
            batch_size=int(args.test_batch_size),
            shuffle=False,
            num_workers=args.workers,
        )
        for ii in range(len(std_test))
    ]
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

        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("BER/train", ber, epoch)
        writer.add_scalar("FER/train", fer, epoch)

        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, "best_model"))
        if epoch % 10 == 0 or epoch in [1, args.epochs]:
            test_loss_list, test_ber_list, test_fer_list = test(
                model, device, test_dataloader_list, EbNo_range_test
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
        "--ebno_syndrome_dB",
        type=float,
        required=True,
        help="Eb/N0 in dB for syndrome",
    )
    parser.add_argument(
        "--ebno_readout_dB",
        type=float,
        required=True,
        help="Eb/N0 in dB for the bits values after measurement",
    )

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
        model_dir = os.path.join(
            "Results_QECCT",
            args.code_type
            + "__Code_n_"
            + str(args.code_n)
            + "_k_"
            + str(args.code_k)
            + "__EbNo_"
            + str(args.ebno_syndrome_dB).replace(".", "_")
            + "__Readout_"
            + str(args.ebno_readout_dB).replace(".", "_")
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
