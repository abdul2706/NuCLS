# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import warnings
from pprint import pprint
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from torch.jit.annotations import Tuple, List, Dict, Optional

from nucls_model.torchvision_detection_utils.utils import save_sample

TAG = '[GeneralizedRCNN.py]'

# noinspection LongLine
def _check_for_degenerate_boxes(trgts):
    if trgts is not None:
        for target_idx, target in enumerate(trgts):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenrate box
                bb_idx = degenerate_boxes.any(dim=1).nonzero().view(-1)[0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError("All bounding boxes should have positive height and width. Found invaid box {} for target at index {}.".format(degen_bb, target_idx))


# noinspection LongLine
class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
        n_testtime_augmentations (int): no oftest-time augmentations.
        proposal_augmenter (RpnProposalAugmenter): this is a function or class that
            can be called to obtain an augmented realization (eg random shift)
            of object proposals from the RPN output.
    """

    def __init__(self, backbone, rpn, roi_heads, transform, proposal_augmenter=None, n_testtime_augmentations=0):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False
        # Mohamed: added this
        if proposal_augmenter is not None:
            assert not roi_heads.batched_nms
        self.n_testtime_augmentations = n_testtime_augmentations
        self.proposal_augmenter = proposal_augmenter

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor of shape [N, 4], got {:}.".format(boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        print(TAG, '[original_image_sizes]', original_image_sizes)
        # print(datetime.now().strftime("%Y-%m-%d %I:%M:%S.%f %p"), TAG, type(images), len(images))
        # print(datetime.now().strftime("%Y-%m-%d %I:%M:%S.%f %p"), TAG, '[images[0]]', type(images[0]), images[0].shape, images[0].min(), images[0].max())
        # save_sample({'img': images[0]}, '04-generalizedrcnn-before-transform')
        images, targets = self.transform(images, targets)
        print(TAG, '[images]', images.image_sizes)
        # print(datetime.now().strftime("%Y-%m-%d %I:%M:%S.%f %p"), TAG, '[images[0]]', type(images[0]), images[0].shape, images[0].min(), images[0].max())
        # print(datetime.now().strftime("%Y-%m-%d %I:%M:%S.%f %p"), TAG, '[images]')
        # print(type(images.tensors), images.tensors.shape)
        # save_sample({'img': images[0]}, '05-generalizedrcnn-after-transform')

        # Check for degenerate boxes
        _check_for_degenerate_boxes(targets)

        features = self.backbone(images.tensors)
        print(TAG, '[features]', features)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        print(TAG, '[proposals]', proposals)
        # print(datetime.now().strftime("%Y-%m-%d %I:%M:%S.%f %p"), TAG, '[proposals]', len(proposals), proposals[0].shape)
        # pprint(proposals)
        # print(datetime.now().strftime("%Y-%m-%d %I:%M:%S.%f %p"), TAG, '[proposal_losses]', proposal_losses)

        _cprobabs = None
        if (not self.training) and (self.n_testtime_augmentations > 0):
            # Mohamed: Test-time augmentation by jittering the RPN output
            #  so that it gets projected onto the feature map multiple times
            #  resulting in slightly different outputs each time. This is a
            #  nice way of augmentation because: 1. The feature map is only
            #  extracted once, only the ROIPooling differs; 2. It augments BOTH
            #  detection and classification.
            # get augmented boxes & get class probabs without postprocessing
            for _ in range(self.n_testtime_augmentations):
                prealization = self.proposal_augmenter(proposals=proposals, image_shapes=images.image_sizes)
                _cprobabs_realization = self.roi_heads(features=features, proposals=prealization, image_shapes=images.image_sizes, _just_return_probabs=True)
                if _cprobabs is None:
                    _cprobabs = _cprobabs_realization
                else:
                    # aggregate soft probabilities
                    _cprobabs += _cprobabs_realization
            _cprobabs = _cprobabs / self.n_testtime_augmentations

        # pass through roi head, possible aggregating probabilities obtained
        # from test-time augmentations
        detections, detector_losses = self.roi_heads(
            features=features, proposals=proposals, _cprobabs=_cprobabs,
            image_shapes=images.image_sizes, targets=targets)
        # print(datetime.now().strftime("%Y-%m-%d %I:%M:%S.%f %p"), TAG, '[detector_losses]', detector_losses)
        print(TAG, '[detections]', detections)

        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        print(TAG, '[detections]', detections)
        # print(datetime.now().strftime("%Y-%m-%d %I:%M:%S.%f %p"), TAG, '[detections]', len(detections), detections[0].keys())
        # print(TAG, '[boxes]', detections[0]['boxes'].shape, detections[0]['boxes'].min(dim=0), detections[0]['boxes'].max(dim=0))
        # print(TAG, '[labels]', detections[0]['labels'].shape)
        # print(TAG, '[masks]', detections[0]['masks'].shape)
        # print(TAG, '[probabs]', detections[0]['probabs'].shape)
        # print(TAG, '[scores]', detections[0]['scores'].shape)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            # noinspection PyRedundantParentheses
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)
