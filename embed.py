import torch

def emb_voc():
    # self.labelmap order: ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
    # 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    # clsid2name = {0: 'aeroplane', 1: 'bike', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat',
    # 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike',
    # 14: 'person', 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor', 20: 'pet', 21: 'cattle', 22: 'wild', 
    # 23: 'people', 24: 'motorized', 25: 'unmotorized', 26: 'utility', 27: 'food container', 28: 'electronics', 29: 'animal',
    # 30: 'vehicles', 31: 'object', 32: 'Root'}

    # clsid2name = {0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat', 4: 'bottle', 5: 'bus', 6: 'car', 7: 'cat',
    # 8: 'chair', 9: 'cow', 10: 'diningtable', 11: 'dog', 12: 'horse', 13: 'motorbike',
    # 14: 'person', 15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tv/monitor', 20: 'pet', 21: 'cattle', 22: 'wild', 
    # 23: 'people', 24: 'motorized', 25: 'unmotorized', 26: 'utility', 27: 'food container', 28: 'electronics', 29: 'animal',
    # 30: 'vehicles', 31: 'object', 32: 'Root'}

    clsid2name = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 
                  5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow', 
                  11: 'dining table', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 
                  16: 'potted plant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tv/monitor'}


    name2clsid = {v: k for k, v in clsid2name.items()}
    # data_voc = {v: k for k, v in torch.load('embeddings/voc_sph_50d.pth').items()}
    data_voc = torch.load('embeddings/voc_sph_50d.pth')
    # data_voc = torch.load('Spherical_embs/voc12_hierarchysph_300d.pth')
    embs_preorder = data_voc['embeddings']
    names_preorder = {v: k for k, v in data_voc['objects'].items()}
    # names_preorder[21] = 'bicycle'
    # x = list(name2clsid.keys())
    # y = sorted(names_preorder)
    # z = list(names_preorder.values())
    # import pdb
    # pdb.set_trace()
    embs = torch.zeros((len(clsid2name), embs_preorder.shape[1]))
    for i, name in enumerate(names_preorder):
        if name in name2clsid:
            embs[name2clsid[name], :] = embs_preorder[i]

    if torch.sum(embs == 0) > 0:
        raise ValueError("Some classes are missing in the embedding file.")

    return embs

# print(emb_voc())

def emb_coco():
    clsid2name = {0: 'background', 1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 
                  5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 
                  11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench', 15: 'bird', 
                  16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow', 21: 'elephant', 22: 'bear', 
                  23: 'zebra', 24: 'giraffe', 25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 
                  29: 'suitcase', 30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 
                  34: 'kite', 35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 
                  39: 'tennis racket', 40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 
                  44: 'knife', 45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich', 
                  50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza', 55: 'donut', 
                  56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant', 60: 'bed', 61: 'dining table', 
                  62: 'toilet', 63: 'tv', 64: 'laptop', 65: 'mouse', 66: 'remote', 67: 'keyboard', 
                  68: 'cell phone', 69: 'microwave', 70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 
                  74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier', 
                  80: 'toothbrush'}

    name2clsid = {v: k for k, v in clsid2name.items()}

    data_coco = torch.load('embeddings/coco_sph_50d.pth')
    embs_preorder = data_coco['embeddings']
    names_preorder = {v: k for k, v in data_coco['objects'].items()}
    # import pdb
    # pdb.set_trace()

    embs = torch.zeros((len(clsid2name), embs_preorder.shape[1]))
    for i, name in enumerate(names_preorder):
        if name in name2clsid:
            embs[name2clsid[name], :] = embs_preorder[i]
 
    if torch.sum(embs == 0) > 0:
        raise ValueError("Some classes are missing in the embedding file.")
    
    return embs

# emb_coco()