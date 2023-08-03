import torch

from models.MLP import MLP


class DETR(torch.nn.Module):
    """
    DETR Model from: <https://arxiv.org/pdf/2005.12872.pdf>
    Code adapted from: <https://github.com/facebookresearch/detr/blob/master/models/detr.py>
    """
    def __init__(self, backbone, transformer, hidden_dim=256, num_classes=41, num_queries=25):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            positional_embedding: torch module of the positional embedding to be used. See positional_embedding.py
            transformer: torch module of the transformer architecture.
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
        """
        super().__init__()
        self.backbone = backbone

        self.num_queries = num_queries
        self.positional_embedding = torch.nn.Embedding(1000, 1)
        self.positional_indices = torch.arange(0, 1000)
        self.query_embedding = torch.nn.Embedding(num_queries, hidden_dim)
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.class_embed = torch.nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, images: torch.Tensor):
        """The forward expects:
               - images: batched images, of shape [batch_size x 3 x H x W]

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        print("images.shape:")
        print(images.shape)

        backbone_output = self.backbone(images)

        print("backbone_output.shape:")
        print(backbone_output.shape)

        positional_embedding = self.positional_embedding(self.positional_indices)
        positional_embedding = positional_embedding.repeat(images.shape[0], 1, 1).flatten(1)

        print("positional_embedding.shape:")
        print(positional_embedding.shape)

        transformer_input = backbone_output + positional_embedding

        hs = self.transformer(src=backbone_output, tgt=self.query_embed())[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
