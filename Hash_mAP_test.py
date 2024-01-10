import torch
import numpy as np
import matplotlib.pyplot as plt
import utils.evaluate as evaluate
import utils.retrieval as retrieval
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from thop import profile, clever_format
import models.AGMH as SEMICON
from AGMH_train import generate_code


def tsne_plot(query_code, query_targets):
    data = query_code.cpu().numpy()
    label = query_targets.argmax(dim=1).numpy()
    tsne = TSNE(n_components=2, init='pca')
    data = tsne.fit_transform(data)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    cls_num = len(set(label))
    color_array = np.random.rand(cls_num, 4)
    color_array[:, -1] = 0.90

    plt.figure()
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], s=5, c=color_array[label[i]].reshape(1, -1))
    plt.xticks([])
    plt.yticks([])
    plt.title('48 bits')
    plt.savefig('cub_48.jpg', dpi=600)


def valid(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args):
    num_classes, att_size, feat_size = args.num_classes, 1, 2048
    model = SEMICON.semicon(code_length=code_length, num_classes=num_classes, att_size=att_size, feat_size=feat_size,
                            device=args.device, pretrained=True)
    model.to(args.device)
    model.load_state_dict(torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/model.pth',
                                     map_location=torch.device('cuda:0')), strict=False)
    model.eval()

    input = torch.randn(1, 3, 224, 224).to(args.device)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], '%.2f')

    # query_code = generate_code(model, query_dataloader, code_length, args)
    query_code = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/query_code.pth')
    B = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/database_code.pth')
    query_dataloader.dataset.get_onehot_targets = torch.load(
        './checkpoints/' + args.info + '/' + str(code_length) + '/query_targets.pth')
    retrieval_targets = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/database_targets.pth')
    # tsne_plot(query_code, query_dataloader.dataset.get_onehot_targets())
    tsne_plot(B, retrieval_targets)
    B = B.to(args.device)
    retrieval_targets = retrieval_targets.to(args.device)

    retrieval.image_retrieval(
        query_dataloader,
        query_code.to(args.device),
        B,
        query_dataloader.dataset.get_onehot_targets().to(args.device),
        retrieval_targets,
        args.root,
        args.dataset,
    )

    mAP = evaluate.mean_average_precision(
        query_code.to(args.device),
        B,
        query_dataloader.dataset.get_onehot_targets().to(args.device),
        retrieval_targets,
        args.device,
        args.topk,
    )
    print("Code_Length: " + str(code_length), end="; ")
    print('[mAP:{:.5f}]'.format(mAP))
