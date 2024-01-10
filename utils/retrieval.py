import os
import torch
import cv2
from tqdm import tqdm


def image_retrieval(query_dataloader, query_code, database_code, query_labels, database_labels,
                    root, dataset):
    """
    Calculate mean average precision(map).

    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.

    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_labels.shape[0]
    wrt_num = 10
    wrt_root = '/home/member/share/luxin/Image Retrieval/GAH/retrieval/' + dataset

    for i in tqdm(range(num_query)):
        query_path = root + query_dataloader.dataset.QUERY_DATA[i]
        query_img = cv2.imread(query_path)
        qimg_name = query_path.split('/')[-1]

        wrt_path = os.path.join(wrt_root, qimg_name)
        cv2.imwrite(wrt_path, query_img)

        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:wrt_num]
        retrieval_res = query_dataloader.dataset.RETRIEVAL_DATA[torch.argsort(hamming_dist).cpu()][:wrt_num]

        for j in range(wrt_num):
            retrieval_path = root + retrieval_res[j]
            retrieval_img = cv2.imread(retrieval_path)

            tof = 'True' if retrieval[j] == 1 else 'False'
            rimg_name = qimg_name.split('.')[0] + '_' + str(j) + '_' + tof + '.jpg'
            wrt_path = os.path.join(wrt_root, rimg_name)
            cv2.imwrite(wrt_path, retrieval_img)
