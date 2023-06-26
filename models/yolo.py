import torch
import torch.nn as nn
from utils import SPP, SAM, BottleneckCSP, Conv
from backbone import resnet18
import numpy as np
import tools
import pmath
import embed
import geoopt

class myYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5, hr=False, mode = 'hyp', open=True):
        super(myYOLO, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 32
        self.grid_cell = self.create_grid(input_size)
        self.input_size = input_size
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()
        self.mode = mode
        self.open = open
        self.number = 17
        # self.embeddings = embed.emb_coco().cuda()
        self.embeddings = embed.emb_voc().cuda()

        # we use resnet18 as backbone
        self.backbone = resnet18(pretrained=True)

        # neck
        self.SPP = nn.Sequential(
            Conv(512, 256, k=1),
            SPP(),
            BottleneckCSP(256*4, 512, n=1, shortcut=False)
        )
        self.SAM = SAM(512)
        self.conv_set = BottleneckCSP(512, 512, n=3, shortcut=False)

        if mode == 'eud':
            self.pred = nn.Conv2d(512, 1 + 50 + 4, 1)
        elif mode == 'hyp':
            self.pred = nn.Conv2d(512, 1 + 50 + 4, 1)
        elif self.mode == 'neck':
            self.ball = geoopt.PoincareBall()
            self.pred = nn.Conv2d(512, 1 + self.number + 4, 1)
        else:
            self.pred = nn.Conv2d(512, 1 + self.number + 4, 1)

    
    def create_grid(self, input_size):
        w, h = input_size[1], input_size[0]
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 2).to(self.device)
        
        return grid_xy

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)
        self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
        self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

    def decode_boxes(self, pred):
        """
        input box :  [tx, ty, tw, th]
        output box : [xmin, ymin, xmax, ymax]
        """
        output = torch.zeros_like(pred)
        pred[:, :, :2] = torch.sigmoid(pred[:, :, :2]) + self.grid_cell
        pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = pred[:, :, 0] * self.stride - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] * self.stride - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] * self.stride + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] * self.stride + pred[:, :, 3] / 2
        
        return output

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def postprocess(self, all_local, all_conf, exchange=True, im_shape=None):
        """
        bbox_pred: (HxW, 4), bsize = 1
        prob_pred: (HxW, num_classes), bsize = 1
        """
        bbox_pred = all_local
        prob_pred = all_conf

        cls_inds = np.argmax(prob_pred, axis=1)
        prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
        scores = prob_pred.copy()
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bbox_pred), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bbox_pred[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bbox_pred = bbox_pred[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        if im_shape != None:
            # clip
            bbox_pred = self.clip_boxes(bbox_pred, im_shape)

        return bbox_pred, scores, cls_inds

    def forward(self, x, target=None):
        # backbone
        _, _, C_5 = self.backbone(x)
        
        # head
        C_5 = self.SPP(C_5)
        C_5 = self.SAM(C_5)
        C_5 = self.conv_set(C_5)

        # pred
        prediction = self.pred(C_5)
        if self.mode == 'eud':
            prediction = prediction.view(C_5.size(0), 1 + 50 + 4, -1).permute(0, 2, 1)
        elif self.mode == 'hyp':
            prediction = prediction.view(C_5.size(0), 1 + 50 + 4, -1).permute(0, 2, 1)
        elif self.mode == 'neck':
            prediction = prediction.view(C_5.size(0), 1 + self.number + 4, -1).permute(0, 2, 1)
        else:
            prediction = prediction.view(C_5.size(0), 1 + self.number + 4, -1).permute(0, 2, 1)

        B, HW, C = prediction.size()

        # Divide prediction to obj_pred, txtytwth_pred and cls_pred   
        # [B, H*W, 1]
        conf_pred = prediction[:, :, :1]
    
        if self.mode == 'eud':
            # [B, H*W, 300]
            cls_pred = prediction[:, :, 1 : 1 + 50]
            # [B, H*W, 4]
            txtytwth_pred = prediction[:, :, 1 + 50:]

            # Generate euclidean random embeddings for 300 dimension
            # cls_embeddings = torch.randn(self.num_classes, self.num_classes).cuda()
            cls_embeddings = self.embeddings
            # and normalize them
            cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1).view(-1,1).clamp_min(1e-12)

        elif self.mode == 'hyp':
            # [B, H*W, 300]
            cls_pred = prediction[:, :, 1 : 1 + 50]
            # [B, H*W, 4]
            txtytwth_pred = prediction[:, :, 1 + 50:]

            # Generate euclidean random embeddings for 300 dimension
            # cls_embeddings = embed.emb_voc().cuda()[:20]

            cls_embeddings = self.embeddings
 
            # if self.open == True:
            #     exclude_indexes = [7, 8, 13]
            #     max_index = len(cls_embeddings[:,1])
            #     cls_embeddings = cls_embeddings[torch.tensor([i for i in range(max_index) if i in exclude_indexes])]
                # cls_embeddings = cls_embeddings[torch.tensor([i for i in range(max_index) if i not in exclude_indexes])]

            # and normalize them
            cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1).view(-1,1).clamp_min(1e-12)

        elif self.mode == 'neck':
            cls_pred = prediction[:, :, 1 : 1 + self.number]
            cls_pred_norm = cls_pred.norm(dim=-3, keepdim=True).clamp_min(1e-15) # dim=-3
            cls_pred = self.ball.expmap0(cls_pred / cls_pred_norm)
            txtytwth_pred = prediction[:, :, 1 + self.number:]
            # cls_embeddings = torch.eye(self.num_classes).cuda()

        else:
            # [B, H*W, num_cls]
            cls_pred = prediction[:, :, 1 : 1 + self.number]
            # [B, H*W, 4]
            txtytwth_pred = prediction[:, :, 1 + self.number:]

            # Generate one-hot encodings
            # cls_embeddings = torch.eye(self.num_classes).cuda()[[7, 8, 13],:]
            # cls_embeddings = torch.randn(self.num_classes, 300).cuda()
            # cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1).view(-1,1).clamp_min(1e-12)


        # test
        if not self.trainable:
            with torch.no_grad():
                # batch size = 1
                all_conf = torch.sigmoid(conf_pred)[0]           # 0 is because that these is only 1 batch.
                all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)

                if self.mode == 'oh':
                    # cls_pred is 1 x 169 x 20 during inference for one hot embeddings.
                    all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_conf)

                elif self.mode == 'eud':
                    # Case 2: Euclidean embeddings + Regression
                    # cls_pred is 1 x 169 x 300 during inference for euclidean embeddings.

                    pw_eud_dist = pmath.pair_wise_eud(cls_pred[0, :, :], cls_embeddings[[7, 8, 13],:]) #[[7, 8, 13],:]
                    class_pred_eud = torch.argmin(pw_eud_dist, dim = 1)
                    all_class = class_pred_eud * all_conf

                elif self.mode == 'hyp':
                    pw_hyp_dist = -pmath.pair_wise_hyp(cls_pred[0, :, :], cls_embeddings[[7, 8, 13],:]) # [[7, 8, 13],:] in open set case, the candidate would be cls_embeddings[[3, 5, 7],:]
                    class_pred_hyp = torch.softmax(pw_hyp_dist, dim = 1)
                    all_class = class_pred_hyp * all_conf
                    # import pdb
                    # pdb.set_trace()
                elif self.mode == 'neck':
                    all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_conf)
                else:
                    # import pdb
                    # pdb.set_trace()
                    all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_conf)

                # separate box pred and class conf
                all_conf = all_conf.to('cpu').numpy()
                all_class = all_class.to('cpu').numpy()
                all_bbox = all_bbox.to('cpu').numpy()
                
                bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)
                # import pdb
                # pdb.set_trace()
                return bboxes, scores, cls_inds
        else:
            if self.mode == 'oh':
                conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                            pred_txtytwth=txtytwth_pred,
                                                                            label=target)
            elif self.mode == 'eud':
                conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss_mse_eud(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                            pred_txtytwth=txtytwth_pred,
                                                                            label=target, cls_embeddings = cls_embeddings)
            elif self.mode == 'hyp':
                conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss_mse_hyp(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                            pred_txtytwth=txtytwth_pred,
                                                                            label=target, cls_embeddings = cls_embeddings)
            elif self.mode == 'neck':
                conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                            pred_txtytwth=txtytwth_pred,
                                                                            label=target)
            else:
                conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                            pred_txtytwth=txtytwth_pred,
                                                                            label=target, cls_embeddings = cls_embeddings)
                
            return conf_loss, cls_loss, txtytwth_loss, total_loss


    # def forward_open(self, x, target=None):
    #     # backbone
    #     _, _, C_5 = self.backbone(x)

    #     # head
    #     C_5 = self.SPP(C_5)
    #     C_5 = self.SAM(C_5)
    #     C_5 = self.conv_set(C_5)

    #     # pred
    #     prediction = self.pred(C_5)
    #     if self.mode == 'eud':
    #         prediction = prediction.view(C_5.size(0), 1 + 300 + 4, -1).permute(0, 2, 1)
    #     elif self.mode == 'hyp':
    #         prediction = prediction.view(C_5.size(0), 1 + 50 + 4, -1).permute(0, 2, 1)
    #     # elif self.mode == 'neck':
    #     #     prediction = prediction.view(C_5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)
    #     else:
    #         prediction = prediction.view(C_5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)
    #     B, HW, C = prediction.size()

    #     # Divide prediction to obj_pred, txtytwth_pred and cls_pred   
    #     # [B, H*W, 1]
    #     conf_pred = prediction[:, :, :1]
    
    #     if self.mode == 'eud':
    #         # [B, H*W, 300]
    #         cls_pred = prediction[:, :, 1 : 1 + 300]
    #         # [B, H*W, 4]
    #         txtytwth_pred = prediction[:, :, 1 + 300:]

    #         # Generate euclidean random embeddings for 300 dimension
    #         # cls_embeddings = embed.emb_voc().cuda()[:20]
    #         cls_embeddings = torch.randn(self.num_classes, 300).cuda()
    #         # and normalize them
    #         cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1).view(-1,1).clamp_min(1e-12)

    #     elif self.mode == 'hyp':
    #         # [B, H*W, 300]
    #         cls_pred = prediction[:, :, 1 : 1 + 50]
    #         # [B, H*W, 4]
    #         txtytwth_pred = prediction[:, :, 1 + 50:]

    #         # Generate euclidean random embeddings for 300 dimension
    #         cls_embeddings = embed.emb_voc().cuda()[:20]
    #         # and normalize them
    #         cls_embeddings = cls_embeddings / cls_embeddings.norm(dim=-1).view(-1,1).clamp_min(1e-12)

    #         # cls_embeddings = embed.emb_coco().cuda()
    #         exclude_indexes = [4, 6, 8]
    #         max_index = 20
    #         cls_embeddings_seen = cls_embeddings[torch.tensor([i for i in range(max_index) if i not in exclude_indexes])]
    #         cls_embeddings_unseen = cls_embeddings[torch.tensor([i for i in range(max_index) if i in exclude_indexes])]

    #     else:
    #         # [B, H*W, num_cls]
    #         cls_pred = prediction[:, :, 1 : 1 + self.num_classes]
    #         # [B, H*W, 4]
    #         txtytwth_pred = prediction[:, :, 1 + self.num_classes:]
    
    #         # Generate one-hot encodings
    #         cls_embeddings = torch.eye(self.num_classes).cuda()
    #         exclude_indexes = [4, 6, 8]
    #         max_index = 20
    #         cls_embeddings_seen = cls_embeddings[torch.tensor([i for i in range(max_index) if i not in exclude_indexes])]
    #         cls_embeddings_unseen = cls_embeddings[torch.tensor([i for i in range(max_index) if i in exclude_indexes])]

    #     # test
    #     if not self.trainable:
    #         with torch.no_grad():
    #             # batch size = 1
    #             all_conf = torch.sigmoid(conf_pred)[0]           # 0 is because that these is only 1 batch.
    #             all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)

    #             if self.mode == 'oh':
    #                 # cls_pred is 1 x 169 x 20 during inference for one hot embeddings.
    #                 all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_conf)

    #             elif self.mode == 'eud':
    #                 # Case 2: Euclidean embeddings + Regression
    #                 # cls_pred is 1 x 169 x 300 during inference for euclidean embeddings.

    #                 pw_eud_dist = pmath.pair_wise_eud(cls_pred[0, :, :], cls_embeddings_unseen)
    #                 class_pred_eud = torch.argmin(pw_eud_dist, dim = 1)
    #                 all_class = class_pred_eud * all_conf

    #             elif self.mode == 'hyp':
    #                 # Case 2: Euclidean embeddings + Regression
    #                 # cls_pred is 1 x 169 x 300 during inference for euclidean embeddings.
    
    #                 pw_hyp_dist = -pmath.pair_wise_hyp(cls_pred[0, :, :], cls_embeddings_unseen) # in open set case, the candidate would be cls_embeddings[[3, 5, 7],:]
    #                 # class_pred_hyp = torch.softmax(pw_hyp_dist, dim = 1)
    #                 class_pred_hyp = torch.argmin(pw_hyp_dist, dim = 1)
    #                 class_pred_hyp = torch.tensor([4,6,8])[class_pred_hyp] # convert 0, 1, 2 to 4, 6, 8.
    #                 all_class = class_pred_hyp * all_conf
    #             # elif self.mode == 'neck':
    #             #     all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_conf)
    #             else:
    #                 all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_conf)

    #             # separate box pred and class conf
    #             all_conf = all_conf.to('cpu').numpy()
    #             all_class = all_class.to('cpu').numpy()
    #             all_bbox = all_bbox.to('cpu').numpy()
                
    #             bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)
  
    #             return bboxes, scores, cls_inds
    #     else:
    #         if self.mode == 'oh':
    #             conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss_mse_oh(pred_conf=conf_pred, pred_cls=cls_pred,
    #                                                                         pred_txtytwth=txtytwth_pred,
    #                                                                         label=target, cls_embeddings = cls_embeddings_seen)
    #         elif self.mode == 'eud':
    #             conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss_mse_eud(pred_conf=conf_pred, pred_cls=cls_pred,
    #                                                                         pred_txtytwth=txtytwth_pred,
    #                                                                         label=target, cls_embeddings = cls_embeddings)
    #         elif self.mode == 'hyp':
    #             conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss_mse_hyp(pred_conf=conf_pred, pred_cls=cls_pred,
    #                                                                         pred_txtytwth=txtytwth_pred,
    #                                                                         label=target, cls_embeddings = cls_embeddings_seen)
    #         # elif self.mode == 'neck':
    #         #     conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
    #         #                                                                 pred_txtytwth=txtytwth_pred,
    #         #                                                                 label=target, cls_embeddings = cls_embeddings)
    #         else:
    #             conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
    #                                                                         pred_txtytwth=txtytwth_pred,
    #                                                                         label=target, cls_embeddings = cls_embeddings_seen)
                
    #         return conf_loss, cls_loss, txtytwth_loss, total_loss



# HYPERBOLIC NECK

# import torch
# import torch.nn as nn
# from utils import SPP, SAM, BottleneckCSP, Conv
# from backbone import resnet18
# import numpy as np
# import pmath
# import embed
# import tools
# import geoopt

# class myYOLO(nn.Module):
#     def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5, hr=False, mode=None):
#         super(myYOLO, self).__init__()
#         self.device = device
#         self.num_classes = num_classes
#         self.trainable = trainable
#         self.conf_thresh = conf_thresh
#         self.nms_thresh = nms_thresh
#         self.stride = 32
#         self.grid_cell = self.create_grid(input_size)
#         self.input_size = input_size
#         self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
#         self.scale_torch = torch.tensor(self.scale.copy(), device=device).float()
#         self.mode = None
#         # we use resnet18 as backbone
#         self.backbone = resnet18(pretrained=True)

#         # neck
#         self.SPP = nn.Sequential(
#             Conv(512, 256, k=1),
#             SPP(),
#             BottleneckCSP(256*4, 512, n=1, shortcut=False)
#         )
#         self.SAM = SAM(512)
#         self.conv_set = BottleneckCSP(512, 512, n=3, shortcut=False)
#         self.ball = geoopt.PoincareBall()

#         self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, 1)
#         # Hyperbolic
#         # self.pred = nn.Conv2d(512, 1 + 300 + 4, 1)
    
#     def create_grid(self, input_size):
#         w, h = input_size[1], input_size[0]
#         # generate grid cells
#         ws, hs = w // self.stride, h // self.stride
#         grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
#         grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
#         grid_xy = grid_xy.view(1, hs*ws, 2).to(self.device)
        
#         return grid_xy

#     def set_grid(self, input_size):
#         self.input_size = input_size
#         self.grid_cell = self.create_grid(input_size)
#         self.scale = np.array([[[input_size[1], input_size[0], input_size[1], input_size[0]]]])
#         self.scale_torch = torch.tensor(self.scale.copy(), device=self.device).float()

#     def decode_boxes(self, pred):
#         """
#         input box :  [tx, ty, tw, th]
#         output box : [xmin, ymin, xmax, ymax]
#         """
#         output = torch.zeros_like(pred)
#         pred[:, :, :2] = torch.sigmoid(pred[:, :, :2]) + self.grid_cell
#         pred[:, :, 2:] = torch.exp(pred[:, :, 2:])

#         # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
#         output[:, :, 0] = pred[:, :, 0] * self.stride - pred[:, :, 2] / 2
#         output[:, :, 1] = pred[:, :, 1] * self.stride - pred[:, :, 3] / 2
#         output[:, :, 2] = pred[:, :, 0] * self.stride + pred[:, :, 2] / 2
#         output[:, :, 3] = pred[:, :, 1] * self.stride + pred[:, :, 3] / 2
        
#         return output

#     def nms(self, dets, scores):
#         """"Pure Python NMS baseline."""
#         x1 = dets[:, 0]  #xmin
#         y1 = dets[:, 1]  #ymin
#         x2 = dets[:, 2]  #xmax
#         y2 = dets[:, 3]  #ymax

#         areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
#         order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

#         keep = []                                             # store the final bounding boxes
#         while order.size > 0:
#             i = order[0]                                      #the index of the bbox with highest confidence
#             keep.append(i)                                    #save it to keep
#             xx1 = np.maximum(x1[i], x1[order[1:]])
#             yy1 = np.maximum(y1[i], y1[order[1:]])
#             xx2 = np.minimum(x2[i], x2[order[1:]])
#             yy2 = np.minimum(y2[i], y2[order[1:]])

#             w = np.maximum(1e-28, xx2 - xx1)
#             h = np.maximum(1e-28, yy2 - yy1)
#             inter = w * h

#             # Cross Area / (bbox + particular area - Cross Area)
#             ovr = inter / (areas[i] + areas[order[1:]] - inter)
#             #reserve all the boundingbox whose ovr less than thresh
#             inds = np.where(ovr <= self.nms_thresh)[0]
#             order = order[inds + 1]

#         return keep

#     def postprocess(self, all_local, all_conf, exchange=True, im_shape=None):
#         """
#         bbox_pred: (HxW, 4), bsize = 1
#         prob_pred: (HxW, num_classes), bsize = 1
#         """
#         bbox_pred = all_local
#         prob_pred = all_conf

#         cls_inds = np.argmax(prob_pred, axis=1)
#         prob_pred = prob_pred[(np.arange(prob_pred.shape[0]), cls_inds)]
#         scores = prob_pred.copy()
        
#         # threshold
#         keep = np.where(scores >= self.conf_thresh)
#         bbox_pred = bbox_pred[keep]
#         scores = scores[keep]
#         cls_inds = cls_inds[keep]

#         # NMS
#         keep = np.zeros(len(bbox_pred), dtype=np.int)
#         for i in range(self.num_classes):
#             inds = np.where(cls_inds == i)[0]
#             if len(inds) == 0:
#                 continue
#             c_bboxes = bbox_pred[inds]
#             c_scores = scores[inds]
#             c_keep = self.nms(c_bboxes, c_scores)
#             keep[inds[c_keep]] = 1

#         keep = np.where(keep > 0)
#         bbox_pred = bbox_pred[keep]
#         scores = scores[keep]
#         cls_inds = cls_inds[keep]

#         if im_shape != None:
#             # clip
#             bbox_pred = self.clip_boxes(bbox_pred, im_shape) 

#         return bbox_pred, scores, cls_inds

#     def forward(self, x, target=None):
#         # backbone
#         _, _, C_5 = self.backbone(x)

#         # head
#         C_5 = self.SPP(C_5)
#         C_5 = self.SAM(C_5)
#         C_5 = self.conv_set(C_5)
#         # import pdb
#         # pdb.set_trace()

#         C_5_norm = C_5.norm(dim=-3, keepdim=True).clamp_min(1e-15) # dim=-3
#         C_5 = self.ball.expmap0(C_5 / C_5_norm)

#         # pred
#         prediction = self.pred(C_5)
#         prediction = prediction.view(C_5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)
#         # Hyperbolic
#         # prediction = prediction.view(C_5.size(0), 1 + 300 + 4, -1).permute(0, 2, 1)
#         B, HW, C = prediction.size()

#         # Divide prediction to obj_pred, txtytwth_pred and cls_pred   
#         # [B, H*W, 1]
#         conf_pred = prediction[:, :, :1]

#         # [B, H*W, num_cls]
#         cls_pred = prediction[:, :, 1 : 1 + self.num_classes]
#         # Hyperbolic
#         # cls_pred = prediction[:, :, 1 : 1 + 300]

#         # hyperbolic neck
#         cls_pred_norm = cls_pred.norm(dim=-3, keepdim=True).clamp_min(1e-15) # dim=-3
#         cls_pred = self.ball.expmap0(cls_pred / cls_pred_norm)

#         # [B, H*W, 4]
#         txtytwth_pred = prediction[:, :, 1 + self.num_classes:]
#         # Hyperbolic
#         # txtytwth_pred = prediction[:, :, 1 + 300:]

#         # test
#         if not self.trainable:
#             with torch.no_grad():
#                 # embeddings = embed.emb_voc().cuda()[:20]
#                 # batch size = 1
#                 all_conf = torch.sigmoid(conf_pred)[0]           # 0 is because that these is only 1 batch.
#                 all_bbox = torch.clamp((self.decode_boxes(txtytwth_pred) / self.scale_torch)[0], 0., 1.)
#                 # Original class
#                 all_class = (torch.softmax(cls_pred[0, :, :], 1) * all_conf)
#                 # Hyperbolic head class
#                 # all_class = (torch.softmax(-pmath.pair_wise_hyp(cls_pred[0,:,:], embeddings, c = 0.1), 1) * all_conf)
                
#                 # separate box pred and class conf
#                 all_conf = all_conf.to('cpu').numpy()
#                 all_class = all_class.to('cpu').numpy()
#                 all_bbox = all_bbox.to('cpu').numpy()

                
#                 bboxes, scores, cls_inds = self.postprocess(all_bbox, all_class)

#                 return bboxes, scores, cls_inds
#         else:
#             conf_loss, cls_loss, txtytwth_loss, total_loss = tools.loss(pred_conf=conf_pred, pred_cls=cls_pred,
#                                                                         pred_txtytwth=txtytwth_pred,
#                                                                         label=target)
#             return conf_loss, cls_loss, txtytwth_loss, total_loss