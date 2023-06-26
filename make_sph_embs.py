import torch
import argparse

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# def similarity(embs):

#     norm = torch.norm(embs, dim=1)  # each row is norm 1.

#     t1 = norm.unsqueeze(1)                                          # n_cls x 1
#     t2 = norm.unsqueeze(0)                                          # 1 x n_cls
#     denominator = torch.matmul(t1, t2)                              # n_cls x n_cls, each element is a norm product
#     numerator = torch.matmul(embs, embs.t())            # each element is a in-prod
#     cos_sim = numerator / denominator                               # n_cls x n_cls, each element is a cos_sim
#     cos_sim_off_diag = cos_sim - torch.diag(torch.diag(cos_sim))
#     obj = cos_sim_off_diag

#     return obj.sum()
    
# parser = argparse.ArgumentParser()
# parser.add_argument('--dim', type=int, required=True)
# parser.add_argument('--lr', type=float, default=1e-4)
# args = parser.parse_args()

# pascal_voc_classes = {
#     0: 'background',
#     1: 'aeroplane',
#     2: 'bicycle',
#     3: 'bird',
#     4: 'boat',
#     5: 'bottle',
#     6: 'bus',
#     7: 'car',
#     8: 'cat',
#     9: 'chair',
#     10: 'cow',
#     11: 'dining table',
#     12: 'dog',
#     13: 'horse',
#     14: 'motorbike',
#     15: 'person',
#     16: 'potted plant',
#     17: 'sheep',
#     18: 'sofa',
#     19: 'train',
#     20: 'tv/monitor'
# }


# if __name__ == '__main__':

#     n_cls = len(pascal_voc_classes)

#     # model = Embedding(n_cls, args.dim).cuda()
#     embs = nn.Parameter(torch.Tensor(n_cls, args.dim))
#     nn.init.normal_(embs, 0, 0.01)

#     loss = 0

#     for iter in range(1000):

#         # normalize embs
#         embs = embs.detach() / torch.norm(embs.detach(), dim=1, keepdim=True)
#         embs = nn.Parameter(embs)
#         optimizer = optim.Adam([embs], args.lr)

#         obj = similarity(embs)
        
#         optimizer.zero_grad()
#         obj.backward()
#         optimizer.step()

#         loss = obj.item()
#         # ema smooth
#         loss_ema = loss if iter == 0 else loss_ema * 0.9 + loss * 0.1

#         if iter % 10 == 0:
#             print('iter', iter, 'loss', loss_ema)


#     embs = embs.detach() / torch.norm(embs.detach(), dim=1, keepdim=True)
#     final_obj = similarity(embs)
#     print('final obj', final_obj.item())
#     outfile = f'voc_sph_{args.dim}d.pth'
#     save_dict = {'objects': pascal_voc_classes, 'embeddings':embs.detach()}
#     torch.save(save_dict, outfile)


# # test code

#     embs = torch.load(f'voc_sph_{args.dim}d.pth')['embeddings']
#     import pdb
#     pdb.set_trace()





def similarity(embs):

    norm = torch.norm(embs, dim=1)  # each row is norm 1.

    t1 = norm.unsqueeze(1)                                          # n_cls x 1
    t2 = norm.unsqueeze(0)                                          # 1 x n_cls
    denominator = torch.matmul(t1, t2)                              # n_cls x n_cls, each element is a norm product
    numerator = torch.matmul(embs, embs.t())            # each element is a in-prod
    cos_sim = numerator / denominator                               # n_cls x n_cls, each element is a cos_sim
    cos_sim_off_diag = cos_sim - torch.diag(torch.diag(cos_sim))
    obj = cos_sim_off_diag

    return obj.sum()
    
parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, required=True)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

ms_coco_classes = {
0: 'background',
1: 'person',
2: 'bicycle',
3: 'car',
4: 'motorcycle',
5: 'airplane',
6: 'bus',
7: 'train',
8: 'truck',
9: 'boat',
10: 'traffic light',
11: 'fire hydrant',
12: 'stop sign',
13: 'parking meter',
14: 'bench',
15: 'bird',
16: 'cat',
17: 'dog',
18: 'horse',
19: 'sheep',
20: 'cow',
21: 'elephant',
22: 'bear',
23: 'zebra',
24: 'giraffe',
25: 'backpack',
26: 'umbrella',
27: 'handbag',
28: 'tie',
29: 'suitcase',
30: 'frisbee',
31: 'skis',
32: 'snowboard',
33: 'sports ball',
34: 'kite',
35: 'baseball bat',
36: 'baseball glove',
37: 'skateboard',
38: 'surfboard',
39: 'tennis racket',
40: 'bottle',
41: 'wine glass',
42: 'cup',
43: 'fork',
44: 'knife',
45: 'spoon',
46: 'bowl',
47: 'banana',
48: 'apple',
49: 'sandwich',
50: 'orange',
51: 'broccoli',
52: 'carrot',
53: 'hot dog',
54: 'pizza',
55: 'donut',
56: 'cake',
57: 'chair',
58: 'couch',
59: 'potted plant',
60: 'bed',
61: 'dining table',
62: 'toilet',
63: 'tv',
64: 'laptop',
65: 'mouse',
66: 'remote',
67: 'keyboard',
68: 'cell phone',
69: 'microwave',
70: 'oven',
71: 'toaster',
72: 'sink',
73: 'refrigerator',
74: 'book',
75: 'clock',
76: 'vase',
77: 'scissors',
78: 'teddy bear',
79: 'hair drier',
80: 'toothbrush'
}


if __name__ == '__main__':

    n_cls = len(ms_coco_classes)

    # model = Embedding(n_cls, args.dim).cuda()
    embs = nn.Parameter(torch.Tensor(n_cls, args.dim))
    nn.init.normal_(embs, 0, 0.01)

    loss = 0

    for iter in range(1000):

        # normalize embs
        embs = embs.detach() / torch.norm(embs.detach(), dim=1, keepdim=True)
        embs = nn.Parameter(embs)
        optimizer = optim.Adam([embs], args.lr)

        obj = similarity(embs)
        
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()

        loss = obj.item()
        # ema smooth
        loss_ema = loss if iter == 0 else loss_ema * 0.9 + loss * 0.1

        if iter % 10 == 0:
            print('iter', iter, 'loss', loss_ema)


    embs = embs.detach() / torch.norm(embs.detach(), dim=1, keepdim=True)
    final_obj = similarity(embs)
    print('final obj', final_obj.item())
    outfile = f'coco_sph_{args.dim}d.pth'
    save_dict = {'objects': ms_coco_classes, 'embeddings':embs.detach()}
    torch.save(save_dict, outfile)


# test code

    embs = torch.load(f'coco_sph_{args.dim}d.pth')['embeddings']
    import pdb
    pdb.set_trace()