import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic
from helper import seg
from utils.box_search import BruteForceBoxSearch, FractionAreaObjective
import clip
from spatial_clip import CLIPMaskedSpatialViT
from spatial_clip import CLIPSpatialResNet
import matplotlib.pyplot as plt



class SLICViT(nn.Module):
    def __init__(self, model='vit14', alpha=0.8, n_segments=[10, 50, 100, 200],
                 aggregation='mean', temperature=1., compactness=50,
                 sigma=0, **args):
        super().__init__()
        if model == 'vit14':
            args['patch_size'] = 14
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'vit16':
            args['patch_size'] = 16
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'vit32':
            args['patch_size'] = 32
            self.model = CLIPMaskedSpatialViT(**args)
        elif model == 'RN50':
            self.model = CLIPSpatialResNet(**args)
        elif model == 'RN50x4':
            self.model = CLIPSpatialResNet(**args)
        else:
            raise Exception('Invalid model name: {}'.format(model))
        self.alpha = alpha
        self.n_segments = n_segments
        self.aggregation = aggregation
        self.temperature = temperature
        self.compactness = compactness
        self.sigma = sigma
        self.window_size = 5

    def get_masks(self, im):
        masks = []
        detection_areas = []
        # Do SLIC with different number of segments so that it has a hierarchical scale structure
        # This can average out spurious activations that happens sometimes when the segments are too small
        for n in self.n_segments:
            # segments_slic = slic(im.astype(
            #     np.float32)/255., n_segments=n, compactness=self.compactness, sigma=self.sigma)
            # print("n:", n)
            # print("segments:",type(segments_slic))
            oct_seg, areas = seg(im.astype(np.float32)/255., n_segments=n, window_size= self.window_size)
            for i in np.unique(oct_seg):
                mask = oct_seg == i
                b_mask = areas[int(i)] == i
                # print(mask)
                masks.append(mask)
                detection_areas.append(b_mask)
        masks = np.stack(masks, 0)
        detection_areas = np.stack(detection_areas, 0)
        return masks, detection_areas

    def get_mask_features(self,im):
        with torch.no_grad():
            # im is uint8 numpy
            h, w = im.shape[:2]
            im = Image.fromarray(im).convert('RGB')
            # im = im.resize((224, 224))
            masks, detection_areas = self.get_masks(np.array(im))
            masks = torch.from_numpy(masks.astype(np.bool)).cuda()
            detection_areas = torch.from_numpy(detection_areas.astype(np.bool)).cuda()
            im = self.model.preprocess(im).unsqueeze(0).cuda()

            image_features = self.model(im, detection_areas)
            image_features = torch.reshape(image_features, (image_features.shape[1], image_features.shape[2]))
            
            print("image_features in clipmap:" , image_features.shape)
            # image_features = torch.permute(image_features,(1, 0))

            image_features = image_features.cpu().float().numpy()
            print("image feature converted to numpy" , image_features.shape)

            return masks.cpu().numpy(), image_features

    def get_mask_scores(self, im, text):
        with torch.no_grad():
            # im is uint8 numpy
            h, w = im.shape[:2]
            im = Image.fromarray(im).convert('RGB')
            im = im.resize((224, 224))
            masks, detection_areas = self.get_masks(np.array(im))
            masks = torch.from_numpy(masks.astype(np.bool)).cuda()
            # masks = torch.from_numpy(masks.astype(np.bool))

            detection_areas = torch.from_numpy(detection_areas.astype(np.bool)).cuda()
            # detection_areas = torch.from_numpy(detection_areas.astype(np.bool))
            im = self.model.preprocess(im).unsqueeze(0).cuda()
            # im = self.model.preprocess(im).unsqueeze(0)
            # print("preprocessed image:", im.shape)

            image_features = self.model(im, detection_areas)
            
            image_features = image_features.permute(0, 2, 1)
            print("features in get heatmap:", image_features.shape)

            text = clip.tokenize([text]).cuda()
            # text = clip.tokenize([text])
            text_features = self.model.encode_text(text)

            image_features = image_features / \
                image_features.norm(dim=1, keepdim=True)
            text_features = text_features / \
                text_features.norm(dim=1, keepdim=True)

            logits = (image_features * text_features.unsqueeze(-1)).sum(1)
            print("logits shape", logits.shape)
            assert logits.size(0) == 1
            logits = logits.cpu().float().numpy()[0]

        return masks.cpu().numpy(), logits

    def get_heatmap(self, im, text):
        masks, logits = self.get_mask_scores(im, text)
        print("masks and logits:", masks.shape, logits.shape)
        heatmap = list(np.nan + np.zeros(masks.shape, dtype=np.float32))
        for i in range(len(masks)):
            mask = masks[i]
            # print("mask:",mask.shape)
            score = logits[i]
            heatmap[i][mask] = score
        heatmap = np.stack(heatmap, 0)
        print("heatmap:", heatmap.shape)

        heatmap = np.exp(heatmap / self.temperature)

        if self.aggregation == 'mean':
            heatmap = np.nanmean(heatmap, 0)
        elif self.aggregation == 'median':
            heatmap = np.nanmedian(heatmap, 0)
        elif self.aggregation == 'max':
            heatmap = np.nanmax(heatmap, 0)
        elif self.aggregation == 'min':
            heatmap = -np.nanmin(heatmap, 0)
        else:
            assert False

        mask_valid = np.logical_not(np.isnan(heatmap))
        _min = heatmap[mask_valid].min()
        _max = heatmap[mask_valid].max()
        heatmap[mask_valid] = (heatmap[mask_valid] -
                               _min) / (_max - _min + 1e-8)
        heatmap[np.logical_not(mask_valid)] = 0.
        return heatmap

    def get_clipmap(self, im, **args):
        _args = {key: getattr(self, key) for key in args}
        for key in args:
            setattr(self, key, args[key])
            print("keys:", key)
        masks, clip_features = self.get_mask_features(im)
        print("mask shape, feature shape:" , masks.shape, clip_features.shape)
        featuremap = (np.nan + np.zeros((masks.shape[1], masks.shape[2],clip_features.shape[1] ), dtype=np.float32))
        # print("featuremap shape", featuremap.shape)
        print("i goes up to:", len(masks))
        for i in range(len(masks)):
            mask = masks[i]
            # print("mask:",mask.shape)
            features = clip_features[i]
            # print("clip features: ", features.shape)
            # print(featuremap.shape)
            featuremap[mask] = features
        featuremap = np.stack(featuremap, 0)
        # print("heatmap:", featuremap.shape)
        return featuremap

    def box_from_heatmap(self, heatmap):
        alpha = self.alpha
        # get accumulated sum map for the objective
        sum_map = heatmap.copy()
        sum_map /= sum_map.sum() + 1e-8
        sum_map -= alpha / sum_map.shape[0] / sum_map.shape[1]
        bf = BruteForceBoxSearch()
        objective = FractionAreaObjective(alpha=alpha)
        box = bf(heatmap, objective)
        box = box.astype(np.float32)[None]
        return box

    def forward(self, im, text, **args):
        # temporary override paramters in init
        _args = {key: getattr(self, key) for key in args}
        for key in args:
            setattr(self, key, args[key])
            print("keys:", key)
        # forward
        h, w = im.shape[:2]
        # im.resize(224,224)
        print("image: ", im.shape)
        heatmap = self.get_heatmap(im, text)
        # print("heatmap is ", type(heatmap), heatmap.shape, heatmap)
        # heatimg = heatmap*800
        # # print(heatimg)
        # o_im = Image.fromarray(im).convert ('RGB')
        # h_im = Image.fromarray(heatimg).convert ('RGB')
        # o_im.save("/gpfs/data/ssrinath/ychen485/implicitSearch/adaptingCLIPtesting/output0/"+text+".png")
        # h_im.save("/gpfs/data/ssrinath/ychen485/implicitSearch/adaptingCLIPtesting/output0/"+text+"_heat.png")
        # print(text+" saved")
        bbox = self.box_from_heatmap(heatmap)
        bbox[:, ::2] = bbox[:, ::2] * w / 224
        bbox[:, 1::2] = bbox[:, 1::2] * h / 224
        # bbox[:, ::2] = bbox[:, ::2] * w / w
        # bbox[:, 1::2] = bbox[:, 1::2] * h / h
        # restore paramters
        for key in args:
            setattr(self, key, _args[key])
        return bbox, heatmap



    def verify(self, image_features_normalized, text, root_path):
        with torch.no_grad():
            r,c,f = image_features_normalized.size()
            input = torch.empty(r, c, 1)
            query_map = torch.zeros_like(input)
            text_tokenized = clip.tokenize([text]).cuda()
            text_features = torch.squeeze(self.model.encode_text(text_tokenized))
            text_features_normalized = text_features
            #text_features_normalized = (text_features - torch.min(text_features)) / (torch.max(text_features) - torch.min(text_features))
            np.save(root_path + "Nesf0_2D/" + text + "_clip_feature", text_features_normalized.cpu().detach().numpy())
            #text_features_normalized = torch.tensor(np.load(root_path + "Nesf0_2D/" + text + "_clip_feature.npy"))
            #print(text_features_normalized)
            text_features_normalized = text_features_normalized.to(torch.float)
            image_features_normalized = image_features_normalized.to(torch.float)
            for i in range(r):
                for j in range(c):
                    query_map[i,j,0] = (torch.dot(image_features_normalized[i,j,:], text_features_normalized) / (np.linalg.norm(image_features_normalized[i,j,:].cpu().detach().numpy()) * np.linalg.norm(text_features_normalized.cpu().detach().numpy())))
        return query_map
