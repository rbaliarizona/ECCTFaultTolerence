import os
import json
import torch
from torch.utils.data import DataLoader
from Main import ECC_Dataset, test, ECC_Transformer, EbN0_to_std
from argparse import Namespace
from Codes import Get_Generator_and_Parity
import logging


def load_args(folder_path):
    args_path = os.path.join(folder_path, "args.json")
    with open(args_path, "r") as f:
        args = json.load(f)
    training_args = Namespace(**args)
    return training_args


def main(folder_path, gpus, workers, ebno_syndrome_dB, ebno_readout_dB):
    # Load args
    training_args = load_args(folder_path)

    # Override ebno_syndrome_dB and ebno_readout_dB if not present in training_args
    if not hasattr(training_args, "ebno_syndrome_dB"):
        logging.info(
            "ebno_syndrome_dB not found in training_args, using provided value."
        )
        training_args.ebno_syndrome_dB = ebno_syndrome_dB
    if not hasattr(training_args, "ebno_readout_dB"):
        logging.info(
            "ebno_readout_dB not found in training_args, using provided value."
        )
        training_args.ebno_readout_dB = ebno_readout_dB

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

    EbNo_range_test = range(4, 7)
    std_test = [EbN0_to_std(ii, code.k / code.n) for ii in EbNo_range_test]
    syndrome_sigma = EbN0_to_std(
        training_args.ebno_syndrome_dB, code.k / code.n
    )
    readout_sigma = EbN0_to_std(training_args.ebno_readout_dB, code.k / code.n)
    test_dataloader_list = [
        DataLoader(
            ECC_Dataset(
                code,
                [std_test[ii]],
                len=int(training_args.test_batch_size),
                syndrome_sigma=syndrome_sigma,
                readout_sigma=readout_sigma,
                zero_cw=False,
            ),
            batch_size=int(training_args.test_batch_size),
            shuffle=False,
            num_workers=workers,
        )
        for ii in range(len(std_test))
    ]
    test(model, device, test_dataloader_list, EbNo_range_test)


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
    parser.add_argument(
        "--ebno_syndrome_dB",
        type=float,
        help="Eb/N0 in dB for syndrome",
    )
    parser.add_argument(
        "--ebno_readout_dB",
        type=float,
        help="Eb/N0 in dB for the bits values after measurement",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(
        args.folder_path,
        args.gpus,
        args.workers,
        args.ebno_syndrome_dB,
        args.ebno_readout_dB,
    )
