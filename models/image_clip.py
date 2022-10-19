import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic
from utils.box_search import BruteForceBoxSearch, FractionAreaObjective
import clip
from spatial_clip import CLIPMaskedSpatialViT
from spatial_clip import CLIPSpatialResNet


class Image_CLIP(nn.Module):
    def __init__(self, model='vit14', alpha=0.8, n_segments=[10, 50, 100, 200],
                 aggregation='mean', temperature=1., compactness=50,
                 sigma=0, **args):
        super().__init__()
        args['patch_size'] = 14
        self.model = CLIPMaskedSpatialViT(**args)
        self.n_segments = n_segments
        self.alpha = alpha
        self.aggregation = aggregation
        self.temperature = temperature
        self.compactness = compactness
        self.sigma = sigma

    def get_masks(self, im):
        masks = []
        for n in self.n_segments: # self.n_segments = [1]
            segments_slic = slic(im.astype(
                np.float32)/255., n_segments=n, compactness=self.compactness, sigma=self.sigma)
            for i in np.unique(segments_slic):
                mask = segments_slic == i
                masks.append(mask)
        masks = np.stack(masks, 0) #(1, 224, 224), All True
        return masks

    def get_image_clip_feature(self, im):
        with torch.no_grad():
            im = Image.fromarray(im).convert('RGB')
            im = im.resize((224, 224))
            masks = self.get_masks(np.array(im))
            masks = torch.from_numpy(masks.astype(np.bool)).cuda()
            im = self.model.preprocess(im).unsqueeze(0).cuda()
            image_features = self.model(im, masks) # same as model.encode_image
            image_features = image_features.permute(0, 2, 1)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features
    
    def verify(self, image_features_normalized, text):
        with torch.no_grad():
            image_features_normalized = torch.squeeze(image_features_normalized)
            text = clip.tokenize([text]).cuda()
            text_features = torch.squeeze(self.model.encode_text(text))
            text_features_normalized = (text_features - torch.min(text_features)) / (torch.max(text_features) - torch.min(text_features))
            logits = torch.dot(image_features_normalized, text_features_normalized) / (np.linalg.norm(image_features_normalized) * np.linalg.norm(text_features_normalized))
            logits = logits.cpu().float().numpy()
        return logits

    def forward(self, im, **args):
        # temporary override paramters in init
        _args = {key: getattr(self, key) for key in args}
        for key in args:
            setattr(self, key, args[key])
            print("keys:", key)
        image_clip_feature = self.get_image_clip_feature(im)
        for key in args:
            setattr(self, key, _args[key])
        return image_clip_feature
