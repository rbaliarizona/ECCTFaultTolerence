import os
import json
import torch
from torch.utils.data import DataLoader
from Main_ECCT_backbone import ECC_Dataset, test, EbN0_to_std, std_to_EbN0
from argparse import Namespace
from Codes import Get_Generator_and_Parity
import logging


def load_args(folder_path):
    args_path = os.path.join(folder_path, "args.json")
    with open(args_path, "r") as f:
        args = json.load(f)
    training_args = Namespace(**args)
    return training_args


def main(folder_path, gpus, workers):
    # Load args
    training_args = load_args(folder_path)
    ebno_syndrome_dBs = [5]
    ebno_readout_dBs = [None]
    training_args.ebno_syndrome_dBs = ebno_syndrome_dBs
    training_args.ebno_readout_dBs = ebno_readout_dBs
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Code:
        pass

    code = Code()
    code.k = training_args.code_k
    code.n = training_args.code_n
    code.code_type = training_args.code_type
    G, H = Get_Generator_and_Parity(
        code, standard_form=training_args.standardize
    )
    code.generator_matrix = torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = torch.from_numpy(H).long()
    training_args.code = code

    # Load model
    model_path = os.path.join(folder_path, "best_model")
    model = torch.load(model_path, map_location=device)

    EbNo_range_test = [6]
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    syndrome_sigmas = [
        EbN0_to_std(ebno_syndrome_dB, code.k / code.n)
        if ebno_syndrome_dB is not None
        else None
        for ebno_syndrome_dB in training_args.ebno_syndrome_dBs
    ]
    readout_sigmas = [
        EbN0_to_std(ebno_readout_dB, code.k / code.n)
        if ebno_readout_dB is not None
        else None
        for ebno_readout_dB in training_args.ebno_readout_dBs
    ]
    test_dataloader_list = [
        DataLoader(
            ECC_Dataset(
                code,
                [std_test[ii]],
                len=int(training_args.test_batch_size),
                syndrome_sigma=[syndrome_sigmas[jj]],
                readout_sigma=[readout_sigmas[kk]],
                zero_cw=False,
            ),
            batch_size=int(training_args.test_batch_size),
            shuffle=False,
            num_workers=workers,
        )
        for ii in range(len(std_test))
        for jj in range(len(syndrome_sigmas))
        for kk in range(len(readout_sigmas))
    ]
    std_to_ebno = lambda std: std_to_EbN0(std, code.k / code.n)  # noqa: E731
    test(model, device, test_dataloader_list, std_to_ebno)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo for ECCT")
    parser.add_argument(
        "-d",
        "--folder_path",
        type=str,
        help="Path to the folder containing the saved model and args.json",
    )
    parser.add_argument("--gpus", type=str, default="-1", help="gpus ids")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    handlers = [
        logging.FileHandler(os.path.join(args.folder_path, 'demo_log.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    main(
        args.folder_path,
        args.gpus,
        args.workers,
    )
